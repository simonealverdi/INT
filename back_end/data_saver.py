# data_saver.py
# Modulo per il salvataggio dei dati nel cluster 'data'

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from memoria import salva_dati_intervista

# Configurazione logging
logger = logging.getLogger(__name__)

# Funzione per salvare una domanda nel nuovo cluster 'data'
def save_question(
    user_id: str,
    session_id: str,
    question_idx: int,
    question_text: str,
    topic: str = None,
    subtopics: List[str] = None,
    keywords: Dict[str, Any] = None
) -> None:
    """Salva una domanda nel cluster 'data'.
    
    Args:
        user_id: ID dell'utente
        session_id: ID della sessione
        question_idx: Indice della domanda
        question_text: Testo della domanda
        topic: Topic principale
        subtopics: Lista dei subtopics
        keywords: Dizionario delle keywords
    """
    try:
        # Salvataggio nel cluster 'data'
        salva_dati_intervista(
            user_id=user_id,
            session_id=session_id,
            question_idx=question_idx,
            domanda=question_text,
            risposta_utente="",  # Prima domanda, non c'è ancora risposta
            topic=topic,
            subtopics=subtopics,
            keywords=keywords
        )
        logger.info(f"Struttura domanda {question_idx} salvata nel cluster 'data' per user_id {user_id}")
    except Exception as e:
        logger.error(f"Errore durante il salvataggio della domanda {question_idx} per user_id {user_id}: {e}")

# Funzione per salvare una risposta dell'utente nel nuovo cluster 'data'
def save_response(
    user_id: str,
    session_id: str,
    question_idx: int,
    question_text: str,
    response_text: str,
    topic: str = None,
    subtopics: List[str] = None,
    keywords: Dict[str, Any] = None,
    non_covered_subtopics: List[str] = None,
    coverage_percent: float = None
) -> None:
    """Salva una risposta dell'utente nel cluster 'data'.
    
    Args:
        user_id: ID dell'utente
        session_id: ID della sessione
        question_idx: Indice della domanda
        question_text: Testo della domanda
        response_text: Testo della risposta dell'utente
        topic: Topic principale
        subtopics: Lista dei subtopics
        keywords: Dizionario delle keywords
        non_covered_subtopics: Lista dei subtopics non coperti nella risposta
        coverage_percent: Percentuale di copertura dei subtopics
    """
    try:
        # Calcola quali subtopics sono stati coperti
        covered_subtopics = []
        if subtopics and non_covered_subtopics:
            covered_subtopics = [sub for sub in subtopics if sub not in non_covered_subtopics]
        
        # Crea struttura per info di copertura
        coverage_info = None
        if subtopics:
            coverage_info = {
                "covered_subtopics": covered_subtopics,
                "non_covered_subtopics": non_covered_subtopics if non_covered_subtopics else [],
                "coverage_percent": coverage_percent if coverage_percent is not None else 100.0
            }
        
        # Salva nel cluster 'data'
        salva_dati_intervista(
            user_id=user_id,
            session_id=session_id,
            question_idx=question_idx,
            domanda=question_text,
            risposta_utente=response_text,
            topic=topic,
            subtopics=subtopics,
            keywords=keywords,
            coverage_info=coverage_info
        )
        logger.info(f"Risposta alla domanda {question_idx} salvata nel cluster 'data' per user_id {user_id}")
    except Exception as e:
        logger.error(f"Errore durante il salvataggio della risposta {question_idx} per user_id {user_id}: {e}")
