from typing import Optional
from Main.core.logger import logger
from Main.core import config
import asyncio, os, base64, hashlib, re, tempfile, time
import boto3
from botocore.exceptions import ClientError

# Import di helper esterni se necessari
import sys
sys.path.append(config.BACK_END_ROOT)
from italian_tts_processor import optimize_italian_tts

# Polly client singleton
_polly_client = None

def get_polly_client():
    """Crea e restituisce un client boto3 per Amazon Polly."""
    global _polly_client
    try:
        # Se il client è già stato inizializzato, restituiscilo
        if _polly_client is not None:
            return _polly_client
            
        # Verifica se le credenziali AWS sono state configurate
        if not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY:
            logger.warning("Credenziali AWS non configurate. Impossibile utilizzare Polly.")
            return None
            
        # Inizializza il client Polly
        _polly_client = boto3.client(
            'polly',
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
        )
        return _polly_client
    except Exception as e:
        logger.error(f"Errore nell'inizializzazione del client AWS Polly: {e}")
        return None

def text_to_speech_polly(text: str, voice_id: str = config.AWS_POLLY_VOICE_ID) -> bytes:
    """Genera audio con Amazon Polly e lo restituisce come bytes.
    
    Args:
        text: Il testo da convertire in audio, può includere tag SSML
        voice_id: L'ID della voce Polly da utilizzare (default: Bianca)
        
    Returns:
        Bytes dell'audio generato
    """
    try:
        # Ottieni il client Polly
        polly_client = get_polly_client()
        if not polly_client:
            logger.error("Client Polly non disponibile. Controlla le credenziali AWS.")
            raise Exception("Client Polly non disponibile")
            
        # Controlla se il testo contiene già tag SSML
        is_ssml = text.strip().startswith('<speak>') and text.strip().endswith('</speak>')
        text_type = 'ssml' if is_ssml else 'text'
        
        # Effettua la richiesta a Polly
        response = polly_client.synthesize_speech(
            Engine=config.AWS_POLLY_ENGINE,
            OutputFormat=config.AWS_POLLY_FORMAT,
            Text=text,
            TextType=text_type,
            VoiceId=voice_id
        )
        
        # Estrai l'audio dalla risposta
        if 'AudioStream' in response:
            return response['AudioStream'].read()
        else:
            logger.error(f"Risposta Polly non contiene AudioStream: {response}")
            raise Exception("Risposta Polly senza audio")
    
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        logger.error(f"AWS Polly error: {error_code} - {error_msg}")
        
        # Errori specifici che possiamo gestire
        if error_code == 'InvalidSsmlException':
            logger.error(f"SSML non valido: {text[:100]}...")
        
        raise Exception(f"Errore AWS Polly: {error_msg}")
        
    except Exception as e:
        logger.error(f"Errore generico con AWS Polly: {e}")
        raise



def text_to_speech(text: str, voice_id: str = config.AWS_POLLY_VOICE_ID, engine: str = config.AWS_POLLY_ENGINE, audio_format: str = config.AWS_POLLY_FORMAT) -> Optional[str]:
    if not text: # Aggiunto controllo per testo vuoto
        logger.warning("text_to_speech chiamato con testo vuoto.")
        return None
    try:
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        cached_audio_path = os.path.join(config.AUDIO_CACHE_DIR, f"{text_hash}.{audio_format}")

        if os.path.exists(cached_audio_path):
            logger.info(f"Audio trovato nella cache: {cached_audio_path}")
            with open(cached_audio_path, "rb") as f:
                audio_content = f.read()
            return base64.b64encode(audio_content).decode('utf-8')

        logger.info(f"Sintesi vocale per: '{text[:50]}...' Voce: {voice_id}")
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat=audio_format,
            VoiceId=voice_id,
            Engine=engine
        )
        audio_content = response['AudioStream'].read()
        
        with open(cached_audio_path, "wb") as f:
            f.write(audio_content)
        logger.info(f"Audio salvato nella cache: {cached_audio_path}")
        
        return base64.b64encode(audio_content).decode('utf-8')
    except Exception as e:
        logger.error(f"Errore durante la sintesi vocale Polly: {e}", exc_info=True)
        return None



async def stream_tts(text: str, voice: str = None):
    """Produce audio con AWS Polly di default; se fallisce, usa OpenAI TTS come fallback."""
    from Main.services.openai_tts_service import text_to_speech_openai
    logger.info(f"TTS Polly (default): richiesta per testo: {text} con voce: {voice if voice else config.AWS_POLLY_VOICE_ID}")

    # --- Tenta Polly (default) ---
    cache_key_polly = f"polly_{text}_{voice}" if voice else f"polly_{text}"
    cached = await get_audio_from_cache(cache_key_polly)
    if cached:
        logger.info(f"TTS Polly: cache HIT per testo: {text[:40]} (voce: {voice if voice else config.AWS_POLLY_VOICE_ID})")
        with open(cached, "rb") as f:
            yield f.read()
        return
    optimized_text = optimize_italian_tts(text, wrap_speak=True)
    logger.info(f"TTS Polly: testo ottimizzato (con SSML): {optimized_text}")
    try:
        selected_voice = voice if voice else config.AWS_POLLY_VOICE_ID
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None, 
            lambda: text_to_speech_polly(optimized_text, selected_voice)
        )
        logger.info(f"TTS Polly: audio generato, lunghezza: {len(audio_bytes) if audio_bytes else 'VUOTO'}")
        if not audio_bytes or len(audio_bytes) < 100:
            raise Exception("TTS Polly: audio non generato o troppo corto!")
        await save_audio_to_cache(cache_key_polly, audio_bytes)
        yield audio_bytes
        return
    except Exception as e:
        logger.error(f"TTS Polly: Errore nella generazione audio: {e}. Si tenta fallback OpenAI TTS...")

    # --- Fallback OpenAI TTS ---
    cache_key_openai = f"openai_{text}_{voice}" if voice else f"openai_{text}"
    cached_openai = await get_audio_from_cache(cache_key_openai)
    if cached_openai:
        logger.info(f"TTS OpenAI: cache HIT per testo: {text[:40]} (voce: {voice if voice else config.OPENAI_TTS_VOICE})")
        with open(cached_openai, "rb") as f:
            yield f.read()
        return
    try:
        selected_voice_openai = voice if voice else config.OPENAI_TTS_VOICE
        audio_bytes_openai = await text_to_speech_openai(
            text=text,
            voice=selected_voice_openai,
            optimize_text=True
        )
        if not audio_bytes_openai or len(audio_bytes_openai) < 100:
            raise Exception("TTS OpenAI: audio non generato o troppo corto!")
        await save_audio_to_cache(cache_key_openai, audio_bytes_openai)
        yield audio_bytes_openai
        return
    except Exception as e:
        logger.error(f"TTS OpenAI: Errore nella generazione audio fallback: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Errore generazione TTS Polly/OpenAI: {str(e)}")



# -----------------------------------------------------------------------------
# Funzione Helper per Generare Struttura Domanda (Topic, Subtopics, Keywords)
# -----------------------------------------------------------------------------



def _wrap_ssml_polly(text: str, speed: float = config.OPENAI_TTS_SPEED) -> str:
    """Wraps text with SSML tags for Polly, including speed control."""
    # Rimuovi le virgolette all'inizio e alla fine che possono causare problemi con SSML
    text = text.strip().strip('"')
    
    # Check if text is already SSML (has <speak> tags)
    is_ssml = text.strip().startswith('<speak>') and text.strip().endswith('</speak>')
    
    # Controlla il valore di velocità
    if not (0.2 <= speed <= 2.0):
        logger.warning(f'Polly speech speed {speed} is outside the typical 0.2-2.0 range. Clamping to 1.0.')
        speed_percentage = '100%'
    else:
        speed_percentage = f"{int(speed * 100)}%"
    
    # Se il testo non ha già tag SSML, aggiungili
    if not is_ssml:
        if speed != 1.0:
            return f"<speak><prosody rate='{speed_percentage}'>{text}</prosody></speak>"
        else:
            return f"<speak>{text}</speak>"
    elif speed != 1.0:
        # Se è già SSML ma vogliamo cambiare la velocità, inseriamo il tag prosody
        # Togliamo i tag <speak></speak> esistenti e li riaggiugiamo con prosody
        text_content = text[7:-8].strip()  # Rimuovi <speak> e </speak>
        
        # Eseguiamo un controllo extra per caratteri speciali problematici per SSML
        text_content = text_content.replace('<', '&lt;').replace('>', '&gt;')
        text_content = text_content.replace('&', '&amp;')
            
        return f"<speak><prosody rate='{speed_percentage}'>{text_content}</prosody></speak>"
    
    # Altrimenti restituisci il testo SSML originale ma con escape dei caratteri speciali
    if is_ssml:
        text_content = text[7:-8].strip()  # Rimuovi <speak> e </speak>
        text_content = text_content.replace('<', '&lt;').replace('>', '&gt;')
        text_content = text_content.replace('&', '&amp;')
        return f"<speak>{text_content}</speak>"
        
    return text



def _wrap_ssml(text: str) -> str:
    """Converte testo plain in SSML con prosodia e pause standard.

    1. Inserisce una breve pausa dopo la punteggiatura forte.
    2. Applica <prosody rate="..%"> per controllare la velocità.

    Args:
        text: Testo già ottimizzato con italian_tts_processor.

    Returns:
        Stringa SSML completa pronta per OpenAI TTS.
    """
    # Aggiunge pause di 300 ms dopo ., ?, ! se non già seguite da tag break
    text_with_breaks = re.sub(r"([.!?])\s+", r"\1 <break time=\"300ms\"/> ", text)
    prosody_rate = int(OPENAI_TTS_SPEED * 100)
    return f"<speak><prosody rate=\"{prosody_rate}%\">{text_with_breaks}</prosody></speak>"

# -----------------------------------------------------------------------------
# Validazione follow-up
# -----------------------------------------------------------------------------


async def get_audio_from_cache(text: str) -> str:
    """Controlla se un testo è già stato sintetizzato e salvato in cache.
    
    Args:
        text: Il testo o la chiave di cache (può includere informazioni sulla voce)
    """
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    cache_path = os.path.join(config.TTS_CACHE_DIR, f"{text_hash}.{config.OPENAI_TTS_FORMAT}")
    
    if os.path.exists(cache_path):
        logger.info(f"TTS cache hit for: {text[:20]}...")
        return cache_path



async def save_audio_to_cache(text: str, audio_bytes: bytes) -> str:
    """Salva l'audio generato in cache.
    
    Args:
        text: Il testo o la chiave di cache (può includere informazioni sulla voce)
        audio_bytes: I byte dell'audio da salvare
    """
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    cache_path = os.path.join(config.TTS_CACHE_DIR, f"{text_hash}.{config.OPENAI_TTS_FORMAT}")
    
    with open(cache_path, "wb") as f:
        f.write(audio_bytes)
    logger.info(f"Saved TTS to cache: {text[:20]}...")
    return cache_path

# COMMENTATO: Funzione originale per lo streaming TTS con OpenAI
# async def stream_tts(text: str, voice: str = None):
#     """Produce audio con OpenAI TTS e caching e logga dettagli per debug.
#     
#     Args:
#         text: Il testo da convertire in audio
#         voice: Opzionale, la voce da utilizzare (se None, usa OPENAI_TTS_VOICE)
#     """
#     logger.info(f"TTS: richiesta per testo: {text} con voce: {voice if voice else OPENAI_TTS_VOICE}")
#     
#     # Usiamo una chiave di cache diversa se viene specificata una voce personalizzata
#     cache_key = f"{text}_{voice}" if voice else text
#     
#     # Controlla la cache
#     cached = await get_audio_from_cache(cache_key)
#     if cached:
#         logger.info(f"TTS: cache HIT per testo: {text[:40]} (voce: {voice if voice else OPENAI_TTS_VOICE})")
#         with open(cached, "rb") as f:
#             yield f.read()
#         return
# 
#     # Pre-processa il testo per migliorare la pronuncia italiana (senza alcun tag SSML)
#     optimized_text = optimize_italian_tts(text, wrap_speak=False) # Assicurati che wrap_speak sia False
#     
#     # Aggiungiamo pause naturali usando punti e spazi invece di tag SSML
#     # Sostituisce virgole, punti e virgola, due punti con una pausa corta
#     optimized_text = re.sub(r"([,;:])\s+", r"\1 . ", optimized_text)
#     # Sostituisce fine frase con una pausa più lunga
#     optimized_text = re.sub(r"([.!?])\s+", r"\1 . . ", optimized_text)
#     
#     # NON utilizziamo alcun tag SSML, solo testo normale
#     logger.info(f"TTS: testo ottimizzato (senza SSML): {optimized_text}")
#     
#     try:
#         # Determina quale voce usare
#         selected_voice = voice if voice else OPENAI_TTS_VOICE
#         
#         # Genera nuovo audio con OpenAI TTS (senza SSML, solo testo normale)
#         resp = client.audio.speech.create(
#             model=OPENAI_TTS_MODEL,  # es. "tts-1-hd"
#             voice=selected_voice,
#             response_format=OPENAI_TTS_FORMAT,
#             speed=OPENAI_TTS_SPEED,
#             input=optimized_text,  # Usiamo il testo ottimizzato direttamente senza SSML
#         )
#         logger.info(f"TTS: risposta ricevuta, tipo: {type(resp)}, content: {hasattr(resp, 'content')}")
#         audio_bytes = resp.content if hasattr(resp, "content") else resp.read()
#         logger.info(f"TTS: lunghezza audio generato: {len(audio_bytes) if audio_bytes else 'VUOTO'}")
#         if not audio_bytes or len(audio_bytes) < 100:
#             logger.error("TTS: audio non generato o troppo corto!")
#             from fastapi import HTTPException
#             raise HTTPException(status_code=500, detail="TTS non riuscito")
#         # Usa la stessa chiave di cache che abbiamo verificato in precedenza
#         await save_audio_to_cache(cache_key, audio_bytes)
#         yield audio_bytes
#     except Exception as e:
#         logger.error(f"TTS: Errore nella generazione audio: {e}")
#         from fastapi import HTTPException
#         raise HTTPException(status_code=500, detail="Errore generazione TTS")


