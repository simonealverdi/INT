from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Dict, Optional
import logging
import base64
import sys
import asyncio
import random
from io import BytesIO

# Import dai moduli interni
from Main.core import config
from Main.models import TTSRequest, TTSResponse, ErrorResponse
from Main.services.tts_service import text_to_speech_polly, get_polly_client
from italian_tts_processor import optimize_italian_tts

# Configurazione logger
logger = logging.getLogger(__name__)

# API router TTS
router = APIRouter(tags=["TTS"])  # Rimuoviamo il prefix dal router perché è già definito in main.py

# File WAV di beep di un secondo (garantito funzionante in tutti i browser)
# Questo contiene un breve tono udibile invece di silenzio
VALID_BEEP_WAV_BASE64 = "UklGRioFAABXQVZFZm10IBAAAAABAAEARKwAABCxAgAECBAAZGF0YQYFAAAAAJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJj///8AAACYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiY"


@router.api_route("/speak", methods=["POST", "GET"], response_model=None, responses={
    500: {"model": ErrorResponse}
})
async def speak(request: TTSRequest = None, voice_id: str = Query(None), text: str = Query(None)):
    # Determina se è una richiesta GET o POST
    is_get_request = request is None and text is not None
    
    # Per le richieste GET, crea un oggetto request dai parametri della query
    if is_get_request:
        request = TTSRequest(text=text, voice_id=voice_id)
    """Converte testo in audio utilizzando l'ottimizzatore italiano e il servizio TTS."""
    # Log di debug
    logger.info(f"=== ENDPOINT /speak - Inizio richiesta TTS ===")
    logger.info(f"Parametri richiesta: voice_id={request.voice_id}, testo='{request.text[:30]}...'")
    logger.info(f"DEVELOPMENT_MODE: {config.DEVELOPMENT_MODE}")
    
    # Voice ID dalla richiesta o predefinito
    voice = request.voice_id or config.AWS_POLLY_VOICE_ID
    logger.info(f"Voice ID scelto: {voice}")
    
    try:
        # STRATEGIA 1: Modalità sviluppo - usa sempre audio fittizio
        if config.DEVELOPMENT_MODE:
            logger.info("Usando audio fittizio per sviluppo")
            # Usa WAV beep (non necessario rimuovere newline)
            audio_base64 = VALID_BEEP_WAV_BASE64
            audio_bytes = base64.b64decode(audio_base64)
            logger.info(f"Audio WAV beep: {len(audio_bytes)} bytes")
            
            # Per richieste GET, restituisce direttamente l'audio come stream binario
            if is_get_request:
                return StreamingResponse(
                    BytesIO(audio_bytes),
                    media_type=f"audio/wav"
                )
            # Per richieste POST, restituisce la risposta JSON con audio_base64
            else:
                return TTSResponse(
                    status="success",
                    message="Audio generato con successo (modalità sviluppo)",
                    audio_base64=audio_base64
                )
        
        # STRATEGIA 2: Produzione - usa Amazon Polly
        logger.info(f"Modalità produzione: generazione audio reale per il testo con voce {voice}")
        
        # Verifica credenziali AWS
        if not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY:
            raise ValueError("Credenziali AWS non configurate")
        
        # Ottimizza il testo per la sintesi vocale italiana se necessario
        final_text = request.text
        is_ssml = final_text.strip().startswith('<speak>') and final_text.strip().endswith('</speak>')
        if not is_ssml:
            final_text = optimize_italian_tts(final_text, wrap_speak=True)
            logger.debug(f"Testo ottimizzato con SSML: {final_text[:100]}...")
        
        # Ottieni client Polly
        polly_client = get_polly_client()
        if not polly_client:
            raise ValueError("Client Polly non disponibile")
        
        # Chiamata a Polly
        logger.info(f"TTS: generazione audio per testo ottimizzato con voce {voice}, motore {config.AWS_POLLY_ENGINE}")
        
        # Ottieni i bytes dell'audio
        audio_bytes = text_to_speech_polly(final_text, voice)
        if not audio_bytes:
            raise ValueError("Nessun audio restituito da AWS Polly")
        
        # Converti in base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        logger.info(f"Audio generato con successo: {len(audio_bytes)} bytes, {len(audio_base64)} chars base64")
        
        # Per le richieste GET, restituisci direttamente lo stream audio
        if is_get_request:
            logger.info(f"Restituisco stream audio binario diretto: {len(audio_bytes)} bytes")
            return StreamingResponse(
                BytesIO(audio_bytes),
                media_type=f"audio/{config.AWS_POLLY_FORMAT}"
            )
        # Per le richieste POST, restituisci la risposta JSON con audio_base64 
        else:
            return TTSResponse(
                status="success",
                message="Audio generato con successo",
                audio_base64=audio_base64
            )
            
    except Exception as e:
        # Gestione errori con audio di fallback
        logger.error(f"Errore durante la sintesi vocale: {e}")
        
        # GESTIONE FALLBACK: In caso di qualsiasi errore, ritorna silenzio valido
        # invece di far fallire completamente la richiesta
        try:
            logger.info("Ritorno audio WAV beep di fallback")
            fallback_audio_base64 = VALID_BEEP_WAV_BASE64
            fallback_audio_bytes = base64.b64decode(fallback_audio_base64)
            
            # Per richieste GET, restituisci direttamente l'audio di fallback
            if is_get_request:
                logger.info(f"GET Fallback: stream audio WAV: {len(fallback_audio_bytes)} bytes")
                return StreamingResponse(
                    BytesIO(fallback_audio_bytes),
                    media_type="audio/wav"
                )
            # Per richieste POST, restituisci JSON con l'audio di fallback
            else:
                logger.info("POST Fallback: JSON con audio_base64")
                return TTSResponse(
                    status="warning",
                    message=f"Usato audio WAV di fallback a causa di un errore: {str(e)}",
                    audio_base64=fallback_audio_base64
                )
        except Exception as fallback_error:
            # Se anche il fallback fallisce, alza un'eccezione
            logger.error(f"Errore anche nel fallback: {fallback_error}")
            raise HTTPException(
                status_code=500, 
                detail=f"Errore durante la sintesi vocale: {str(e)}. Fallback fallito: {str(fallback_error)}"
            )

@router.get("/status", response_model=TTSResponse)
async def tts_status() -> TTSResponse:
    """Verifica lo stato del servizio TTS"""
    
    # Verifica delle credenziali e configurazioni disponibili
    has_aws = bool(config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY)
    
    if config.DEVELOPMENT_MODE and not has_aws:
        service_type = "development"
        description = "Servizio TTS in modalità sviluppo (mock data)"
    elif has_aws:
        service_type = "aws_polly"
        description = f"AWS Polly con voce {config.AWS_POLLY_VOICE_ID}, motore {config.AWS_POLLY_ENGINE}"
    else:
        service_type = "unavailable"
        description = "Nessun servizio TTS configurato (AWS Polly)"
    
    # Include informazioni sulla cache TTS
    import os
    cache_dir = config.TTS_CACHE_DIR
    cache_files = len(os.listdir(cache_dir)) if os.path.exists(cache_dir) else 0
    
    return TTSResponse(
        status="success",
        message=f"Servizio TTS {service_type} disponibile | Cache files: {cache_files}",
        audio_base64="" # Campo vuoto per l'endpoint di status
    )


@router.get("/available_voices")
async def available_voices():
    """Restituisce l'elenco delle voci disponibili per il TTS."""
    try:
        # Lista di voci disponibili di Amazon Polly
        voices = [
            # Voci italiane
            {"id": "Bianca", "name": "Bianca (Amazon)", "lang": "it-IT", "provider": "amazon"},
            {"id": "Carla", "name": "Carla (Amazon)", "lang": "it-IT", "provider": "amazon"},
            {"id": "Giorgio", "name": "Giorgio (Amazon)", "lang": "it-IT", "provider": "amazon"},
            # Voci inglesi
            {"id": "Joanna", "name": "Joanna (Amazon)", "lang": "en-US", "provider": "amazon"},
            {"id": "Matthew", "name": "Matthew (Amazon)", "lang": "en-US", "provider": "amazon"},
            {"id": "Amy", "name": "Amy (Amazon)", "lang": "en-GB", "provider": "amazon"},
            {"id": "Emma", "name": "Emma (Amazon)", "lang": "en-GB", "provider": "amazon"}
        ]
        
        logger.info(f"Disponibili {len(voices)} voci per TTS")
        return {"voices": voices, "default": "bianca"}
    except Exception as e:
        logger.error(f"Errore nel recupero delle voci disponibili: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Errore nel recupero delle voci disponibili: {str(e)}"
        )

@router.post("/stream", response_class=StreamingResponse)
async def stream_tts(request: TTSRequest):
    """Converte testo in audio e lo restituisce come stream di byte.
    
    Questo endpoint è particolarmente utile per file audio più grandi,
    in quanto permette di iniziare la riproduzione mentre l'audio viene ancora caricato.
    """
    logger.info(f"Richiesta di stream audio per il testo: '{request.text[:50]}...'")
    
    try:
        # Voice ID può essere fornito dalla richiesta o preso dalla configurazione
        voice = request.voice_id or config.AWS_POLLY_VOICE_ID
        
        # In modalità sviluppo senza credenziali AWS, genera un file audio semplice
        if config.DEVELOPMENT_MODE and (not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY):
            # Creiamo un file temporaneo con dati mock
            logger.info("Modalità sviluppo: streaming di dati audio simulati")
            audio_bytes = b"MOCK_AUDIO_DATA_FOR_DEVELOPMENT" * 100  # Ripetuto per dimensioni maggiori
            return StreamingResponse(
                BytesIO(audio_bytes),
                media_type=f"audio/{config.AWS_POLLY_FORMAT}"
            )

        # Genera audio con Amazon Polly
        # Ottimizza il testo se necessario
        final_text = request.text
        is_ssml = final_text.strip().startswith('<speak>') and final_text.strip().endswith('</speak>')
        if not is_ssml:
            final_text = optimize_italian_tts(final_text, wrap_speak=True)
            
        audio_bytes = text_to_speech_polly(
            text=final_text,
            voice_id=voice
        )
        
        if not audio_bytes:
            logger.error("Nessun dato audio generato")
            raise HTTPException(status_code=500, detail="Errore nella generazione dell'audio")
        
        # Crea uno streaming response direttamente dai byte
        return StreamingResponse(
            BytesIO(audio_bytes),
            media_type=f"audio/{config.AWS_POLLY_FORMAT}"
        )
        
    except Exception as e:
        logger.error(f"Errore durante la sintesi vocale in streaming: {e}")
        raise HTTPException(status_code=500, detail=f"Errore durante la sintesi vocale: {str(e)}")
