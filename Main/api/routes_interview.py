from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, List, Any, Optional
import logging
import uuid
from datetime import datetime

# Import dai moduli interni
from Main.core import config
from Main.models import (
    InterviewResponse, ErrorResponse, InterviewStatusResponse,
    InterviewQuestionResponse, InterviewResultResponse, AnswerRequest
)
from Main.application.interview_state_adapter import (
    get_state, reset_session, get_current_question, 
    save_answer, advance_to_next_question
)
from .auth import get_current_user

# Configurazione logger
logger = logging.getLogger(__name__)
# Router per l'intervista
router = APIRouter(tags=["Interview"])

@router.post("/start", response_model=InterviewResponse, responses={
    401: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def start_interview(request: Request, current_user: str = Depends(get_current_user)) -> InterviewResponse:
    """Avvia una nuova sessione di intervista"""
    try:
        # Utilizza l'ID utente ottenuto dal token JWT o le credenziali fisse
        user_id = current_user  # current_user è già l'ID utente (stringa)
        
        # Azzera eventuali sessioni precedenti per questo utente
        reset_session(user_id)
        
        # Ottieni una nuova sessione di intervista per l'utente
        session = get_state(user_id)
        interview_id = session.session_id
        
        logger.info(f"Nuova intervista avviata: {interview_id} per utente: {user_id}")
        
        return InterviewResponse(
            status="success", 
            message="Intervista avviata con successo", 
            interview_id=interview_id
        )
        
    except Exception as e:
        logger.error(f"Errore durante l'avvio dell'intervista: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Errore durante l'avvio dell'intervista: {str(e)}"
        )

@router.get("/next-question/{interview_id}", response_model=InterviewQuestionResponse, responses={
    404: {"model": ErrorResponse},
    410: {"model": InterviewResponse}
})
async def get_next_question(interview_id: str) -> Dict[str, Any]:
    """Ottieni la prossima domanda per un'intervista"""
    # Cerca l'utente associato all'ID dell'intervista
    # Per semplicità in questa versione, l'ID dell'intervista è l'ID dell'utente
    user_id = config.DEV_USERNAME  # Default per sviluppo
    
    # Ottieni lo stato dell'intervista
    session = get_state(user_id)
    
    # Verifica se l'intervista è quella richiesta
    if session.session_id != interview_id:
        logger.warning(f"Intervista non trovata: {interview_id}")
        raise HTTPException(status_code=404, detail="Intervista non trovata")
    
    # Verifica se tutte le domande sono state poste
    all_questions_asked = False
    if hasattr(session, 'questions_asked') and hasattr(session, 'questions'):
        all_questions_asked = len(session.questions_asked) >= len(session.questions)

    # Se tutte le domande sono state poste, restituisci un messaggio di completamento
    if all_questions_asked:
        # Segna l'intervista come completata
        session.completed = True
        # Restituisci una risposta di completamento invece di un errore
        return InterviewQuestionResponse(
            status="completed",
            message="Intervista completata",
            interview_id=interview_id,
            question={
                "id": "interview_completed",
                "text": "Grazie per aver partecipato all'intervista.",
                "type": "completion",
                "interview_completed": True
            }
    )

    # Verifica se l'intervista è già completata
    if session.completed:
        return InterviewResponse(
            status="completed",
            message="L'intervista è già stata completata",
            interview_id=interview_id
        )
    
    # Ottieni la prossima domanda dall'intervista
    question = get_current_question(session)
    
    # Verifica se la domanda corrente è una domanda di follow-up
    is_follow_up = hasattr(session, 'current_question_is_follow_up_for_subtopic') and session.current_question_is_follow_up_for_subtopic is not None
    
    # Se la domanda è un follow-up, aggiungi metadati specifici
    if is_follow_up:
        if "type" not in question:
            question["type"] = "follow_up"
        if "follow_up_for" not in question and session.current_question_is_follow_up_for_subtopic:
            question["follow_up_for"] = session.current_question_is_follow_up_for_subtopic
        logger.info(f"Servita domanda di follow-up per intervista {interview_id}, topic: {session.current_question_is_follow_up_for_subtopic}")
    
    # Se non ci sono più domande disponibili, segna l'intervista come completata
    if not question or "id" not in question:
        # Tutte le domande sono state poste, segna l'intervista come completata
        session.completed = True
        raise HTTPException(
            status_code=410,  # Gone - risorsa non più disponibile 
            detail="Tutte le domande sono state poste. L'intervista è completata."
        )
    
    # Restituisci la domanda corrente
    return InterviewQuestionResponse(
        status="success",
        message="Domanda disponibile",
        interview_id=interview_id,
        question=question
    )

@router.post("/submit-answer/{interview_id}/{question_id}", response_model=InterviewResponse, responses={
    404: {"model": ErrorResponse},
    400: {"model": ErrorResponse}
})
async def submit_answer(interview_id: str, question_id: str, answer: AnswerRequest) -> InterviewResponse:
    """Invia una risposta a una domanda dell'intervista"""
    # Recupera l'utente associato all'intervista
    user_id = config.DEV_USERNAME  # Default per sviluppo
    
    # Ottieni lo stato dell'intervista
    session = get_state(user_id)
    
    # Verifica se l'intervista è quella richiesta
    if session.session_id != interview_id:
        logger.warning(f"Intervista non trovata: {interview_id}")
        raise HTTPException(status_code=404, detail="Intervista non trovata")
    
    # Verifica se l'intervista è già completata
    if session.completed:
        return InterviewResponse(
            status="error",
            message="L'intervista è già stata completata",
            interview_id=interview_id
        )
    
    # Verifica che la domanda sia quella corrente
    current_question = get_current_question(session)
    current_question_id = current_question.get("id") if current_question else None
    
    if current_question_id != question_id:
        logger.warning(f"ID domanda non corrisponde: atteso {current_question_id}, ricevuto {question_id}")
        return InterviewResponse(
            status="error",
            message="ID domanda non corrisponde alla domanda corrente",
            interview_id=interview_id
        )
    
    # Salva la risposta utilizzando l'adapter
    needed_followup, missing_topics, coverage = save_answer(session, answer.answer_text)
    
    logger.info(f"Risposta registrata per intervista {interview_id}, domanda {question_id}")
    
    # Se non sono necessari follow-up, possiamo avanzare alla domanda successiva
    if not needed_followup:
        advance_to_next_question(session)
    
    return InterviewResponse(
        status="success",
        message="Risposta registrata con successo",
        interview_id=interview_id
    )

@router.get("/status/{interview_id}", response_model=InterviewStatusResponse, responses={
    404: {"model": ErrorResponse}
})
async def get_interview_status(interview_id: str) -> InterviewStatusResponse:
    """Ottieni lo stato attuale di un'intervista"""
    # Recupera l'utente associato all'intervista
    user_id = config.DEV_USERNAME  # Default per sviluppo
    
    # Ottieni lo stato dell'intervista
    session = get_state(user_id)
    
    # Verifica se l'intervista è quella richiesta
    if session.session_id != interview_id:
        logger.warning(f"Intervista non trovata: {interview_id}")
        raise HTTPException(status_code=404, detail="Intervista non trovata")
    
    # Ottieni la domanda corrente
    current_question = get_current_question(session)
    current_question_id = current_question.get("id") if current_question else None
    
    return InterviewStatusResponse(
        status="success",
        message="Stato intervista recuperato",
        interview_id=interview_id,
        user_id=session.user_id,
        start_time=session.start_time,
        current_question_id=current_question_id,
        questions_asked=session.questions_asked,
        answers_count=len(session.answers),
        completed=session.completed,
        score=session.score
    )

# Aggiungiamo AnswerRequest che è stato rimosso ma è ancora utilizzato
from Main.models import AnswerRequest

@router.post("/end/{interview_id}", response_model=InterviewResultResponse, responses={
    404: {"model": ErrorResponse}
})
async def end_interview(interview_id: str) -> InterviewResultResponse:
    """Termina un'intervista in corso"""
    # Recupera l'utente associato all'intervista
    user_id = config.DEV_USERNAME  # Default per sviluppo
    
    # Ottieni lo stato dell'intervista
    session = get_state(user_id)
    
    # Verifica se l'intervista è quella richiesta
    if session.session_id != interview_id:
        logger.warning(f"Intervista non trovata: {interview_id}")
        raise HTTPException(status_code=404, detail="Intervista non trovata")
    
    # Contrassegna l'intervista come completata e calcola un punteggio
    session.completed = True
    
    # Assegna un punteggio fittizio (in una versione reale, questo verrebbe calcolato in base alle risposte)
    import random
    session.score = random.randint(60, 100)
    
    # In una versione reale, qui salveremmo il risultato finale nel database
    if not config.DEVELOPMENT_MODE and config.MONGODB_ENABLED:
        try:
            # Potremmo usare una funzione specifica per salvare il risultato finale
            logger.info(f"Dati finali dell'intervista {interview_id} salvati nel database")
        except Exception as e:
            logger.error(f"Errore nel salvataggio dei dati finali: {e}")
    
    logger.info(f"Intervista {interview_id} terminata con punteggio: {session.score}")
    
    return InterviewResultResponse(
        status="success",
        message="Intervista terminata con successo",
        interview_id=interview_id,
        score=session.score,
        questions_asked=len(session.questions_asked),
        answers_provided=len(session.answers)
    )
