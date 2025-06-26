from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends, Query, Cookie, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import logging
import base64
import os
import uuid

# Import dai moduli interni
from Main.core import config
from .auth import get_current_user_optional
from Main.application.user_session_service import get_state
import urllib.parse

# Configurazione logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API router
router = APIRouter(tags=["Transcribe"])

@router.post("/transcribe", response_model=Dict[str, Any])
async def transcribe_audio(
    audio: UploadFile = File(...),
    user_id: str = Form(...),
    audio_only: bool = Form(False),
    token_query: Optional[str] = Form(None)
):
    """
    Riceve un file audio di risposta, simula una trascrizione (per ora), e restituisce la prossima domanda come audio.
    """
    try:
        logger.info(f"Ricevuto file audio da utente: {user_id}, dimensione: {audio.size} bytes")
        
        # Trascriviamo l'audio usando il servizio OpenAI Whisper
        try:
            # Leggiamo il contenuto del file audio
            audio_content = await audio.read()
            logger.debug(f"Audio ricevuto: {len(audio_content)} bytes")
            
            # Importiamo il servizio di trascrizione
            from Main.services.whisper_service import speech_to_text_openai
            
            # Verifichiamo se siamo in modalità di sviluppo
            if config.DEVELOPMENT_MODE:
                logger.warning("DEVELOPMENT_MODE attivo: utilizzo trascrizione simulata")
                transcription = "Risposta simulata dell'utente"
            else:
                logger.info("Trascrizione con OpenAI Whisper in corso...")
                transcription = await speech_to_text_openai(audio_content)
                logger.info(f"Trascrizione completata: {transcription[:100]}...")
        except Exception as e:
            logger.error(f"Errore durante la trascrizione: {e}", exc_info=True)
            # Fallback alla trascrizione simulata in caso di errore
            transcription = "Risposta simulata (errore trascrizione)"
            logger.warning(f"Utilizzata trascrizione fallback: {transcription}")
        
        # Recupera la sessione dell'utente
        session = get_state(user_id)
        logger.info(f"Sessione recuperata per {user_id}: {session.interview_id}")
        
        # Salviamo la risposta dell'utente
        from Main.application.interview_state_adapter import save_answer, advance_to_next_question
        
        # Aumento del livello di log globale per debug
        import logging
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)
        
        try:
            # Salviamo la risposta e verifichiamo se è necessario un follow-up
            # NOTA: save_answer internamente avanza già alla prossima domanda se necessario
            # e restituisce i valori in quest'ordine: (needs_followup, coverage_percent, missing_topics)
            needed_followup, coverage, missing_topics = save_answer(session, transcription)
            logger.info(f"Analisi risposta: needed_followup={needed_followup}, coverage={coverage:.1f}%, missing_topics={missing_topics}")
            
            # Verifichiamo i metadati della domanda corrente
            if hasattr(session, 'current_topic') and session.current_topic:
                logger.debug(f"Metadati domanda: topic={session.current_topic}, subtopics={session.current_subtopics}")
                if hasattr(session, 'current_keywords') and session.current_keywords:
                    logger.debug(f"Keywords domanda: {session.current_keywords}")
                    
            # Se è richiesto un follow-up, la domanda successiva sarà di tipo follow-up
            # altrimenti procediamo con la domanda successiva dello script
            if needed_followup:
                follow_up_subtopic = getattr(session, 'current_question_is_follow_up_for_subtopic', 'attributo non disponibile')
                logger.info(f"Domanda di follow-up necessaria per subtopic: {follow_up_subtopic}")
            else:
                logger.info("Nessun follow-up necessario, procedo con la prossima domanda")
                # NOTA: L'avanzamento alla prossima domanda ora viene gestito internamente
                # nella funzione save_answer, quindi non è più necessario chiamare next_question qui
        finally:
            # Ripristiniamo il livello di log precedente
            root_logger.setLevel(previous_level)
        
        # Ottieni la prossima domanda (se c'è)
        next_question = None
        
        # Otteniamo la domanda corrente dalla sessione
        from Main.application.interview_state_adapter import get_current_question
        
        # Recuperiamo i dati della domanda corrente
        try:
            # Verifichiamo se la domanda è di tipo follow-up
            is_follow_up = hasattr(session, 'current_question_is_follow_up_for_subtopic') and session.current_question_is_follow_up_for_subtopic is not None
            
            # Recuperiamo la domanda corrente (che può essere una domanda principale o un follow-up)
            question_data = get_current_question(session)
            logger.debug(f"Question data recuperati: {question_data}")
            
            if question_data and 'text' in question_data:
                next_question = {
                    'id': question_data.get('id', 'q1'),
                    'Domanda': question_data.get('text', ''),
                    'is_follow_up': is_follow_up
                }
                logger.info(f"Prossima domanda ({'FOLLOW-UP' if is_follow_up else 'PRINCIPALE'}): {next_question['Domanda'][:50]}...")
                
                if is_follow_up:
                    subtopic = getattr(session, 'current_question_is_follow_up_for_subtopic', 'attributo non disponibile')
                    logger.info(f"Domanda di follow-up per subtopic: {subtopic}")
            else:
                logger.warning("Nessuna domanda disponibile dalla sessione")
                next_question = None
        except Exception as e:
            logger.error(f"Errore nel recupero della domanda corrente: {e}")
            next_question = None
        
        # Se non ci sono domande disponibili, usa una risposta generica
        if not next_question:
            logger.warning(f"Nessuna domanda disponibile per l'utente: {user_id}")
            generic_response = "Non ci sono altre domande disponibili. L'intervista è terminata."
            
            # Non generiamo più l'audio direttamente qui, utilizziamo solo l'endpoint TTS interno
            voice = "Bianca"  # Usa la voce italiana predefinita per il TTS interno
            logger.info(f"Utilizzo TTS interno per il testo: '{generic_response[:50]}...' con voce {voice}")
            
            # Non generiamo più audio qui, solo l'URL per il TTS
            audio_content = ""  # Nessun audio embedded
            
            # Prepariamo l'URL per il TTS con i parametri necessari
            encoded_text = urllib.parse.quote(generic_response)
            tts_url = f"/api/tts/speak?voice_id={voice}&text={encoded_text}"
            
            return {
                "status": "end",
                "message": "Intervista terminata",
                "audio_content": audio_content,
                "text": generic_response,
                "audio_url": tts_url,
                "type": "end"
            }
        
        # Altrimenti, prepara la prossima domanda
        question_text = next_question.get('Domanda', 'Non ci sono altre domande disponibili.')
        
        # Non generiamo più l'audio direttamente qui, utilizziamo solo l'endpoint TTS interno
        voice = "Bianca"  # Usa la voce italiana predefinita per il TTS interno
        logger.info(f"Utilizzo TTS interno per il testo: '{question_text[:50]}...' con voce {voice}")
        
        # Non generiamo audio qui
        audio_content = ""  # Nessun audio embedded
        
        # Prepariamo l'URL per il TTS con i parametri necessari
        encoded_text = urllib.parse.quote(question_text)
        tts_url = f"/api/tts/speak?voice_id={voice}&text={encoded_text}"
        
        # Prepara la risposta
        response = {
            "status": "success",
            "message": "Audio processato con successo",
            "transcription": transcription,
            "audio_content": audio_content,
            "text": question_text,
            "audio_url": tts_url,
            "type": "question",
            "question_id": next_question.get('id', '')
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Errore nella trascrizione dell'audio: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Errore nella trascrizione: {str(e)}"
        )
