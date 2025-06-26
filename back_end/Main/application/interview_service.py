from typing import Dict, List, Any, Optional
import logging
from Main.application.user_session_service import DOMANDE

# Import dall'adapter che utilizza la classe InterviewState esterna
from Main.application.interview_state_adapter import (
    get_state, 
    get_current_question, 
    save_answer, 
    advance_to_next_question
)
# Non importiamo InterviewState direttamente per evitare cicli di importazione

# Configurazione logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



async def _process_next_step(session: Any, response_text: Optional[str] = None) -> Dict[str, Any]:
    """Elabora la risposta dell'utente e determina la prossima domanda dell'intervista."""
    logger.info(f"Elaborazione prossimo passo per session idx={session.idx}, risposta={response_text[:50] if response_text else 'Nessuna'}...")
    
    # Salva lo stato precedente per possibile utilizzo (followup mantiene topic/subtopics)
    previous_topic = session.current_topic
    previous_subtopics = session.current_subtopics
    previous_keywords = session.current_keywords
    
    # Salva eventuale risposta dell'utente per la domanda corrente
    if response_text and session.idx < len(session.script):
        current_q_data = session.script[session.idx]
        current_q_text = current_q_data.get("Domanda", "")
        
        # Salva la risposta dell'utente nei database
        try:
            salva_dati_intervista(
                user_id=session.user_id,
                session_id=str(session.idx),  # MODIFICATO: Usa str(session.idx) come session_id
                question_idx=session.idx,
                domanda=current_q_text,
                risposta_utente=response_text,
                topic=session.current_topic,
                subtopics=session.current_subtopics,
                keywords=session.current_keywords
            )
            logger.info(f"Risposta utente salvata nel database per user_id {session.user_id}, q_idx {session.idx}")
        except Exception as e:
            logger.error(f"Errore nel salvare la risposta utente: {e}")
    
    # Avanza all'indice della domanda successiva
    session.idx += 1
    
    # Verifica se abbiamo finito le domande
    if session.idx >= len(session.script):
        logger.info(f"Intervista completata: tutte le {len(session.script)} domande sono state esaurite")
        return {
            "message": "Colloquio completato.",
            "question_text": None, 
            "question_type": "completed",
            "question_index": session.idx,
            "current_topic": session.current_topic,
            "current_subtopics": session.current_subtopics,
            "current_keywords": session.current_keywords
        }
    
    # Ottieni la prossima domanda dallo script
    next_question_data = session.script[session.idx]
    next_question_text = next_question_data.get("Domanda", "")
    next_question_type = next_question_data.get("Tipologia", "main")  # Default a 'main' se non specificato
    
    if not next_question_text:
        logger.error(f"Testo della prossima domanda mancante allo script index {session.idx}")
        return {
            "message": "Errore: testo della domanda mancante.",
            "question_text": None,
            "question_type": "error",
            "question_index": session.idx,
            "current_topic": session.current_topic,
            "current_subtopics": session.current_subtopics,
            "current_keywords": session.current_keywords
        }
    
    # Genera topic, subtopics, e keywords SOLO per le domande principali
    if next_question_type == "main":
        logger.info(f"Generazione struttura per domanda principale: '{next_question_text[:50]}...' ")
        try:
            topic, subtopics, keywords = await generate_question_structure(session, next_question_text)
            
            session.current_topic = topic
            session.current_subtopics = subtopics
            session.current_keywords = keywords if keywords else []  # Ripristina a [] per coerenza
            logger.info(f"Struttura generata: Topic='{topic}', Subtopics={subtopics}, Keywords presenti={'sì' if keywords else 'no'}")
            
        except Exception as e:
            logger.error(f"Errore generazione struttura per domanda {next_question_text[:30]}: {e}", exc_info=True)
            # Assicura stato pulito in caso di errore
            session.current_topic = None
            session.current_subtopics = []
            session.current_keywords = []
    else:  # Domanda di tipo 'followup' dallo script
        logger.info(f"Nuova domanda è 'followup' dallo script. Manteniamo topic/subtopics precedenti se presenti.")
        # Mantiene topic/subtopic/keyword precedenti (se la domanda 'main' li aveva generati)
        session.current_topic = previous_topic
        session.current_subtopics = previous_subtopics
        session.current_keywords = previous_keywords
    
    # --- Preparazione Risultato ---
    result_payload = {
        "message": "Prossima domanda.",
        "question_text": next_question_text,
        "question_type": next_question_type,
        "question_index": session.idx,  # L'indice logico corrente
        "current_topic": session.current_topic,
        "current_subtopics": session.current_subtopics,
        "current_keywords": session.current_keywords
    }
    
    logger.debug(f"Exiting _process_next_step with payload: {result_payload}")
    return result_payload

async def handle_empty_transcription(session: Any, question_text: str) -> Dict[str, Any]:
    """Gestisce il caso di trascrizione vuota generando una richiesta di chiarimento.
    Versione semplificata per sviluppo."""
    logger.info(f"Gestione trascrizione vuota per la domanda: '{question_text[:30]}...'.")
    
    # Prepara una risposta standard senza chiamare API esterne
    clarification_message = "Non ho ricevuto alcuna risposta. Potresti rispondere alla domanda, per favore?"
    
    return {
        "status": "clarification_needed",
        "message": clarification_message,
        "question_text": question_text,
        "question_type": "clarification"
    }



async def generate_question_structure(session: Any, question_text: str) -> tuple:
    
    
        logger.info(f"Analisi struttura domanda: '{question_text[:50]}...'")  
    
        try:
            # Cerca la domanda in DOMANDE
            matching_question = next((q for q in DOMANDE if q['testo'] == question_text), None)
        
            if matching_question and matching_question.get('topic') and matching_question.get('subtopics') and matching_question.get('keywords'):
                # Estrai i metadati dalla domanda trovata
                topic = matching_question['topic']
                subtopics = matching_question['subtopics']
                keywords = matching_question['keywords']  # Già nel formato lista di liste
            
                logger.info(f"Metadati LLM trovati: Topic={topic}, Subtopics={len(subtopics)}, Keywords={len(keywords)} liste")
                return topic, subtopics, keywords  # Restituisce direttamente la lista di liste
            
            if matching_question:
                logger.warning(f"Metadati incompleti per la domanda trovata")
        except Exception as e:
            logger.error(f"Errore nell'accesso ai metadati: {e}")
    
    # Fallback con metadati semplificati
        logger.warning("Utilizzo metadati di fallback")
    
        topic = "Argomento generico"
        subtopics = ["Aspetto 1", "Aspetto 2", "Aspetto 3"]
        keywords = [  # Lista di liste direttamente
            ["parola1", "parola2", "parola3", "parola4", "parola5"],
        ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
        ["termine1", "termine2", "termine3", "termine4", "termine5"]
    ]
    
        logger.info(f"Struttura di fallback generata: Topic={topic}, Subtopics={subtopics}, Keywords={len(keywords)} liste")
        return topic, subtopics, keywords
