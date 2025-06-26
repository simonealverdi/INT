"""
Servizio di persistenza dati che integra i moduli esterni memoria.py e data_saver.py.

Questo servizio fornisce un'API semplificata per salvare e recuperare dati relativi alle interviste,
utilizzando il cluster 'data' di MongoDB in produzione e un fallback in memoria per lo sviluppo.
"""

from typing import Dict, List, Any, Optional, Union
import logging
import json
import os
from datetime import datetime

from Main.core import config
from data_saver import save_question, save_response
from memoria import salva_dati_intervista

# Configurazione logger
logger = logging.getLogger(__name__)

# Storage in memoria per modalità sviluppo
_dev_storage: Dict[str, Any] = {
    "interviews": [],
    "questions": [],
    "responses": []
}

def save_interview_question(
    user_id: str,
    session_id: str, 
    question_idx: int,
    question_text: str,
    topic: str = None,
    subtopics: List[str] = None,
    keywords: Dict[str, Any] = None
) -> bool:
    """
    Salva una domanda di intervista nel database o in memoria se in modalità sviluppo.
    
    Args:
        user_id: ID dell'utente
        session_id: ID della sessione di intervista
        question_idx: Indice della domanda
        question_text: Testo della domanda
        topic: Topic principale della domanda (opzionale)
        subtopics: Lista dei subtopics (opzionale)
        keywords: Dizionario delle keywords (opzionale)
        
    Returns:
        True se il salvataggio è avvenuto con successo
    """
    try:
        if config.DEVELOPMENT_MODE or not config.MONGODB_ENABLED:
            # In modalità sviluppo, salva in memoria
            _dev_storage["questions"].append({
                "user_id": user_id,
                "session_id": session_id,
                "question_idx": question_idx,
                "question_text": question_text,
                "topic": topic,
                "subtopics": subtopics,
                "keywords": keywords,
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"Domanda {question_idx} salvata in memoria (modalità sviluppo)")
            return True
        else:
            # In modalità produzione, utilizza data_saver.py
            save_question(
                user_id=user_id,
                session_id=session_id,
                question_idx=question_idx,
                question_text=question_text,
                topic=topic,
                subtopics=subtopics,
                keywords=keywords
            )
            logger.info(f"Domanda {question_idx} salvata nel database per utente {user_id}")
            return True
    except Exception as e:
        logger.error(f"Errore durante il salvataggio della domanda: {e}")
        return False

def save_interview_response(
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
) -> bool:
    """
    Salva una risposta dell'utente nel database o in memoria se in modalità sviluppo.
    
    Args:
        user_id: ID dell'utente
        session_id: ID della sessione di intervista
        question_idx: Indice della domanda
        question_text: Testo della domanda
        response_text: Testo della risposta dell'utente
        topic: Topic principale (opzionale)
        subtopics: Lista dei subtopics (opzionale)
        keywords: Dizionario delle keywords (opzionale)
        non_covered_subtopics: Lista dei subtopics non coperti (opzionale)
        coverage_percent: Percentuale di copertura dei subtopics (opzionale)
        
    Returns:
        True se il salvataggio è avvenuto con successo
    """
    try:
        if config.DEVELOPMENT_MODE or not config.MONGODB_ENABLED:
            # In modalità sviluppo, salva in memoria
            _dev_storage["responses"].append({
                "user_id": user_id,
                "session_id": session_id,
                "question_idx": question_idx,
                "question_text": question_text,
                "response_text": response_text,
                "topic": topic,
                "subtopics": subtopics,
                "keywords": keywords,
                "non_covered_subtopics": non_covered_subtopics,
                "coverage_percent": coverage_percent,
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"Risposta alla domanda {question_idx} salvata in memoria (modalità sviluppo)")
            return True
        else:
            # In modalità produzione, utilizza data_saver.py
            save_response(
                user_id=user_id,
                session_id=session_id,
                question_idx=question_idx,
                question_text=question_text,
                response_text=response_text,
                topic=topic,
                subtopics=subtopics,
                keywords=keywords,
                non_covered_subtopics=non_covered_subtopics,
                coverage_percent=coverage_percent
            )
            logger.info(f"Risposta alla domanda {question_idx} salvata nel database per utente {user_id}")
            return True
    except Exception as e:
        logger.error(f"Errore durante il salvataggio della risposta: {e}")
        return False

def save_interview_result(
    user_id: str,
    session_id: str,
    score: int,
    questions_count: int,
    answers_count: int
) -> bool:
    """
    Salva il risultato finale di un'intervista nel database o in memoria se in modalità sviluppo.
    
    Args:
        user_id: ID dell'utente
        session_id: ID della sessione di intervista
        score: Punteggio finale
        questions_count: Numero di domande poste
        answers_count: Numero di risposte fornite
        
    Returns:
        True se il salvataggio è avvenuto con successo
    """
    try:
        if config.DEVELOPMENT_MODE or not config.MONGODB_ENABLED:
            # In modalità sviluppo, salva in memoria
            _dev_storage["interviews"].append({
                "user_id": user_id,
                "session_id": session_id,
                "score": score,
                "questions_count": questions_count,
                "answers_count": answers_count,
                "completed": True,
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"Risultato dell'intervista {session_id} salvato in memoria (modalità sviluppo)")
            return True
        else:
            # In modalità produzione, utilizza memoria.py direttamente
            salva_dati_intervista(
                user_id=user_id,
                session_id=session_id,
                question_idx=-1,  # -1 indica che è il risultato finale
                domanda="",
                risposta_utente="",
                topic="intervista_completata",
                coverage_info={
                    "score": score,
                    "questions_count": questions_count,
                    "answers_count": answers_count
                }
            )
            logger.info(f"Risultato dell'intervista {session_id} salvato nel database per utente {user_id}")
            return True
    except Exception as e:
        logger.error(f"Errore durante il salvataggio del risultato: {e}")
        return False

def dump_dev_storage_to_file() -> bool:
    """
    Esporta i dati di sviluppo in un file JSON se in modalità sviluppo.
    
    Returns:
        True se l'esportazione è avvenuta con successo
    """
    if not config.DEVELOPMENT_MODE:
        logger.warning("dump_dev_storage_to_file chiamato fuori dalla modalità sviluppo")
        return False
        
    try:
        # Assicurati che esista la directory
        dump_dir = os.path.join(config.BACK_END_ROOT, "data_dumps")
        os.makedirs(dump_dir, exist_ok=True)
        
        # Nome del file con timestamp
        filename = f"dev_storage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(dump_dir, filename)
        
        # Salva i dati in un file JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(_dev_storage, f, ensure_ascii=False, indent=2, default=str)
            
        logger.info(f"Dati di sviluppo esportati in {filepath}")
        return True
    except Exception as e:
        logger.error(f"Errore durante l'esportazione dei dati di sviluppo: {e}")
        return False
