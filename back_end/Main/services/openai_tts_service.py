"""
Servizio per la sintesi vocale utilizzando l'API OpenAI TTS.
Integra il processore di testo italiano per migliorare la qualità della sintesi.
"""

import os
import hashlib
import logging
import asyncio
import random  # Necessario per generare audio fittizio
from typing import Optional, Union
import openai
import tempfile
from pathlib import Path

from Main.core import config
from italian_tts_processor import optimize_italian_tts

# Configurazione del logger
logger = logging.getLogger(__name__)


def get_openai_client():
    """
    Crea e restituisce un client OpenAI.
    
    Returns:
        Client OpenAI o None se non configurato
    """
    try:
        if not config.OPENAI_API_KEY:
            logger.warning("OpenAI API Key non configurata")
            return None
            
        return openai.OpenAI(api_key=config.OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Errore nell'inizializzazione del client OpenAI: {e}")
        return None


async def text_to_speech_openai(
    text: str, 
    voice: str = None, 
    model: str = None, 
    format: str = None,
    speed: float = None,
    optimize_text: bool = True
) -> Optional[bytes]:
    """
    Genera audio con OpenAI TTS e lo restituisce come bytes.
    
    Args:
        text: Testo da convertire in audio
        voice: Voce da utilizzare (default da config)
        model: Modello TTS da utilizzare (default da config)
        format: Formato audio (default da config)
        speed: Velocità dell'audio (default da config) 
        optimize_text: Se ottimizzare il testo per l'italiano
        
    Returns:
        Bytes dell'audio generato o None in caso di errore
    """
    try:
        # Verifica se siamo in modalità sviluppo
        if config.DEVELOPMENT_MODE:
            logger.info("=== TTS: Modalità sviluppo attiva - generazione audio fittizio ===")
            # Genera un audio fittizio valido per il test (formato MP3 minimale)
            # Questa è la testata di un file MP3 valido
            mp3_header = bytes.fromhex("494433030000000000004C414D4503")
            # Aggiungi dati fittizi ma in quantità sufficiente a essere un file MP3 valido
            mock_audio = mp3_header + bytes([random.randint(0, 255) for _ in range(10000)])
            logger.info(f"Audio fittizio generato: {len(mock_audio)} bytes")
            return mock_audio
            
        # Ottieni client OpenAI
        client = get_openai_client()
        if not client:
            logger.error("Client OpenAI non disponibile")
            # In modalità sviluppo, ritorna audio fittizio anche se manca l'API key
            if config.DEVELOPMENT_MODE:
                logger.warning("Usando audio simulato per lo sviluppo")
                return b"\x00\x01\x02\x03\x04\x05" * 1000
            return None
            
        # Usa i parametri dalla configurazione se non specificati
        model = model or config.OPENAI_TTS_MODEL
        voice = voice or config.OPENAI_TTS_VOICE
        format = format or config.OPENAI_TTS_FORMAT
        speed = speed if speed is not None else config.OPENAI_TTS_SPEED
        
        # Ottimizza il testo per la sintesi vocale italiana
        final_text = text
        if optimize_text:
            is_ssml = text.strip().startswith('<speak>') and text.strip().endswith('</speak>')
            if not is_ssml:
                # Ottimizza il testo e aggiungi tag SSML
                final_text = optimize_italian_tts(text, wrap_speak=True)
                logger.debug(f"Testo ottimizzato con SSML: {final_text[:100]}...")
            else:
                # Il testo è già in formato SSML
                logger.debug("Il testo contiene già tag SSML, nessuna ottimizzazione necessaria")
                
        # Controllo cache
        cache_key = f"{text}_{voice}_{model}_{speed}_{format}"
        cache_file = get_audio_from_cache(cache_key)
        if cache_file:
            logger.info(f"TTS: Audio trovato in cache per: {text[:30]}...")
            with open(cache_file, "rb") as f:
                return f.read()
        
        # Chiamata asincrona a OpenAI TTS
        logger.info(f"TTS: generazione audio per: {text[:50]}... con voce {voice}, modello {model}")
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=final_text,
            response_format=format,
            speed=speed
        )
        
        # Estrai audio dalla risposta
        audio_bytes = response.content
        
        # Salva in cache per uso futuro
        await save_audio_to_cache(cache_key, audio_bytes)
        
        return audio_bytes
    
    except Exception as e:
        logger.error(f"Errore durante la generazione audio con OpenAI TTS: {e}")
        return None


def get_audio_from_cache(cache_key: str) -> Optional[str]:
    """
    Controlla se un testo è già stato sintetizzato e salvato in cache.
    
    Args:
        cache_key: La chiave di cache che identifica l'audio
        
    Returns:
        Il percorso del file in cache o None se non trovato
    """
    key_hash = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
    cache_path = os.path.join(config.TTS_CACHE_DIR, f"{key_hash}.{config.OPENAI_TTS_FORMAT}")
    
    if os.path.exists(cache_path):
        return cache_path
    return None


async def save_audio_to_cache(cache_key: str, audio_bytes: bytes) -> Optional[str]:
    """
    Salva l'audio generato in cache.
    
    Args:
        cache_key: La chiave di cache che identifica l'audio
        audio_bytes: I byte dell'audio da salvare
        
    Returns:
        Il percorso del file in cache o None in caso di errore
    """
    try:
        key_hash = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
        cache_path = os.path.join(config.TTS_CACHE_DIR, f"{key_hash}.{config.OPENAI_TTS_FORMAT}")
        
        # Assicurati che la directory esista
        os.makedirs(config.TTS_CACHE_DIR, exist_ok=True)
        
        # Salva in cache
        with open(cache_path, "wb") as f:
            f.write(audio_bytes)
            
        logger.info(f"Audio salvato in cache: {cache_path}")
        return cache_path
    except Exception as e:
        logger.error(f"Errore durante il salvataggio in cache: {e}")
        return None
        

async def stream_audio_file(file_path: str, chunk_size: int = 1024):
    """
    Funzione di supporto per lo streaming di file audio.
    
    Args:
        file_path: Percorso del file audio
        chunk_size: Dimensione del chunk per lo streaming
    """
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk
            await asyncio.sleep(0.01)  # Piccola pausa per evitare blocchi
