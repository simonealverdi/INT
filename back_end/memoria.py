
# memoria.py
# Modulo aggiornato per salvare i dati dell'intervista nel cluster 'data'

import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pymongo import MongoClient, errors
from pymongo.server_api import ServerApi

# ------------------------------------------------------------------ #
# 1.  Carica la stringa di connessione una sola volta
# ------------------------------------------------------------------ #
load_dotenv()                                    # legge .env

@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    """Restituisce (e memo­rizza) l'oggetto MongoClient."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("Variabile d'ambiente MONGODB_URI mancante")
    return MongoClient(uri, server_api=ServerApi("1"))

# connessione e db sono ora disponibili a livello di modulo
try:
    client = get_client()
    client.admin.command("ping")                 # test rapido
    db = client["data"]  # Utilizziamo solo il cluster 'data'
except errors.PyMongoError as exc:               # connessione fallita
    print(f"❌  Errore di connessione a MongoDB: {exc}")
    db = None                                    # evita NameError


# ------------------------------------------------------------------ #
# 2.  Funzione per salvare i dati dell'intervista nel cluster 'data'
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
# 3.  Test manuale: esegui `python memoria.py`
# ------------------------------------------------------------------ #
def salva_dati_intervista(
    user_id: str,
    session_id: str,
    question_idx: int,
    domanda: str,
    risposta_utente: str,
    topic: str = None,
    subtopics: list = None,
    keywords: dict = None,
    coverage_info: dict = None
) -> None:
    """
    Salva i dati strutturati dell'intervista nel cluster 'data'.
    Ogni domanda e risposta viene salvata come documento separato.
    
    Args:
        user_id: ID dell'utente
        session_id: ID della sessione di intervista
        question_idx: Indice della domanda nell'intervista
        domanda: Testo della domanda
        risposta_utente: Testo della risposta dell'utente
        topic: Argomento principale della domanda
        subtopics: Lista dei sottotopics associati alla domanda
        keywords: Dizionario di keywords per topic e subtopics
        coverage_info: Informazioni sulla copertura dei subtopics
    """
    if db is None:
        raise RuntimeError("Connessione MongoDB non inizializzata")
    
    # Preparazione dei dati in una struttura pulita
    documento = {
        "user_id": user_id,
        "session_id": session_id,
        "question_idx": question_idx,
        "domanda": domanda,
        "risposta_utente": risposta_utente,
        "timestamp": datetime.now(timezone.utc)
    }
    
    # Aggiungiamo topic e subtopics se presenti
    if topic:
        documento["topic"] = topic
    
    if subtopics:
        documento["subtopics"] = subtopics
        
    if keywords:
        documento["keywords"] = keywords
        
    if coverage_info:
        documento["coverage_info"] = coverage_info
    
    # Salvataggio nel database
    db["interviste"].insert_one(documento)

# ------------------------------------------------------------------ #
# 3.  Test manuale: esegui `python memoria.py`
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("✅  Connessione a MongoDB riuscita")
    # Test del salvataggio nel cluster 'data'
    salva_dati_intervista(
        user_id="test_user",
        session_id="test_session",
        question_idx=0,
        domanda="Qual è il tuo film preferito?",
        risposta_utente="Mi piace molto Inception.",
        topic="Preferenze cinematografiche",
        subtopics=["generi preferiti", "registi preferiti"],
        keywords={
            "topic_keywords": ["cinema", "film", "preferenze"],
            "subtopic_keywords": {
                "generi preferiti": ["azione", "fantascienza"],
                "registi preferiti": ["nomi", "stili"]
            }
        },
        coverage_info={
            "covered_subtopics": ["generi preferiti"],
            "non_covered_subtopics": ["registi preferiti"],
            "coverage_percent": 50.0
        }
    )
    print("Documento salvato nel cluster 'data'.")