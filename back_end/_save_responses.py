# _save_responses.py
# Script temporaneo per testare il salvataggio dei dati nel nuovo cluster 'data'

import json
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from memoria import salva_memoria, salva_dati_intervista

# Funzione di test per salvare una domanda
def test_save_question(
    user_id: str,
    session_id: str,
    question_idx: int,
    question_text: str,
    topic: str = None,
    subtopics: List[str] = None,
    keywords: Dict[str, Any] = None
):
    """Testa il salvataggio di una domanda nel nuovo cluster 'data'."""
    print(f"\nSalvataggio domanda nel cluster 'data':\n  User: {user_id}\n  Session: {session_id}\n  Index: {question_idx}")
    
    # Per retrocompatibilità, salva anche nel formato vecchio
    if topic:
        memoria_contenuto = {
            "question_text": question_text,
            "topic": topic,
            "subtopics": subtopics,
            "keywords": keywords
        }
        # Conversione in JSON
        testo_memoria = json.dumps(memoria_contenuto)
        # Salva nel vecchio formato
        salva_memoria(user_id=user_id, testo=testo_memoria, keyword=[])
        print("  ✓ Salvato nel formato vecchio (retrocompatibilità)")
    
    # Salva nel nuovo formato strutturato
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
    print("  ✓ Salvato nel nuovo formato strutturato")

# Funzione di test per salvare una risposta dell'utente
def test_save_response(
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
):
    """Testa il salvataggio di una risposta dell'utente nel nuovo cluster 'data'."""
    print(f"\nSalvataggio risposta nel cluster 'data':\n  User: {user_id}\n  Session: {session_id}\n  Index: {question_idx}")
    
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
    
    # Per retrocompatibilità, salva anche nel formato vecchio
    if topic:
        memoria_contenuto = {
            "question_text": question_text,
            "response_text": response_text,
            "topic": topic,
            "subtopics": subtopics,
            "keywords": keywords,
            "coverage_info": coverage_info
        }
        # Conversione in JSON
        testo_memoria = json.dumps(memoria_contenuto)
        # Salva nel vecchio formato
        salva_memoria(user_id=user_id, testo=testo_memoria, keyword=[])
        print("  ✓ Salvato nel formato vecchio (retrocompatibilità)")
    
    # Salva nel nuovo formato strutturato
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
    print("  ✓ Salvato nel nuovo formato strutturato")

# Test principale
if __name__ == "__main__":
    # Test con una domanda
    test_save_question(
        user_id="test_user",
        session_id="test_session_123",
        question_idx=0,
        question_text="Parlami della tua famiglia",
        topic="Famiglia",
        subtopics=["numero totale di membri della famiglia", "tipologia di relazioni familiari", "attività condivise"],
        keywords={
            "topic_keywords": ["famiglia", "parenti", "casa"],
            "subtopic_keywords": {
                "numero totale di membri della famiglia": ["quanti", "numero", "persone"],
                "tipologia di relazioni familiari": ["rapporti", "relazioni", "legami"],
                "attività condivise": ["attività", "insieme", "condivise"]
            }
        }
    )
    
    # Test con una risposta dell'utente
    test_save_response(
        user_id="test_user",
        session_id="test_session_123",
        question_idx=0,
        question_text="Parlami della tua famiglia",
        response_text="Nella mia famiglia siamo in 4: io, mia moglie e i nostri due figli. Facciamo spesso attività insieme come andare in bicicletta o guardare film.",
        topic="Famiglia",
        subtopics=["numero totale di membri della famiglia", "tipologia di relazioni familiari", "attività condivise"],
        keywords={
            "topic_keywords": ["famiglia", "parenti", "casa"],
            "subtopic_keywords": {
                "numero totale di membri della famiglia": ["quanti", "numero", "persone"],
                "tipologia di relazioni familiari": ["rapporti", "relazioni", "legami"],
                "attività condivise": ["attività", "insieme", "condivise"]
            }
        },
        non_covered_subtopics=["tipologia di relazioni familiari"],
        coverage_percent=66.7
    )
    
    print("\nTest completato con successo!")
