"""Topic Detection utilities
=========================
Un unico modulo che raggruppa **tutto** ciò che serve alla nuova pipeline:

* arricchimento metadati in fase *offline* (già implementato nella sezione
  `QuestionImporter`);
* soglie letta da **variabili d’ambiente** (`TH_FUZZY`, `TH_COS`,
  `COVERAGE_THRESHOLD_PERCENT`) così l’utente può ritoccarle via `.env` senza
  toccare il codice;
* funzione `detect_covered_topics()` usata a *runtime* da
  `InterviewState.missing_topics`;
* piccola helper `topic_objects_from_meta()` che prende i campi salvati nello
  YAML e costruisce una lista di oggetti `Topic` pronta per la cascata.

L’idea è che **interview_state.py** debba solo:
```python
from topic_detection import topic_objects_from_meta, detect_covered_topics
```
poi, dentro `missing_topics()`:
```python
# 0. init
topics = topic_objects_from_meta(
            self.current_subtopics,
            self.current_keywords,   # lista‑di‑liste, già allineata
            self.current_lemma_sets, # nuovo campo nel DOMANDE
            self.current_fuzzy_norms,
            self.current_vectors,
        )
covered, coverage = detect_covered_topics(last_user_text, topics)
missing  = [t.name for t in topics if t.name not in covered]
return missing, coverage*100
```
Così non cambiamo nomi delle variabili già presenti – aggiungiamo solo i
nuovi array paralleli `lemma_sets`, `fuzzy_norms`, `vectors` nella variabile
`DOMANDE`.
"""

from __future__ import annotations

import json
import os
import re
import logging
import time
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Any, Tuple, Set

import numpy as np
import unidecode  # Modifica: importiamo il modulo invece della funzione

# ---------------------------------------------------------------------------
# Configurazione del servizio NLP
# ---------------------------------------------------------------------------
NLP_URL = os.getenv("NLP_URL", "http://127.0.0.1:9000/parse")
logging.info(f"Servizio NLP configurato su: {NLP_URL}")

# ---------------------------------------------------------------------------
# Third‑party libs (import lazy per evitare crash se mancano SBERT / RapidFuzz)
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    from rapidfuzz.fuzz import token_sort_ratio  # type: ignore
except ImportError:  # pragma: no cover
    token_sort_ratio = None  # type: ignore

# ---------------------------------------------------------------------------
# NLP models (ora gestiti dal servizio NLP)
# ---------------------------------------------------------------------------
# Il modello spaCy è ora gestito dal servizio NLP

SBERT = None
if SentenceTransformer:
    try:
        SBERT = SentenceTransformer("paraphrase-multilingual-MiniLM-L6-v2")
    except Exception as exc:  # pragma: no cover
        logging.warning(f"SBERT non caricato, fallback al servizio NLP: {exc}")

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soglie da .env (con fallback a default robusti)
# ---------------------------------------------------------------------------
TH_FUZZY: int = int(os.getenv("TH_FUZZY", "90"))  # 0‑100
TH_COS: float = float(os.getenv("TH_COS", "0.75"))
COVERAGE_THRESHOLD_PERCENT: float = float(os.getenv("COVERAGE_THRESHOLD_PERCENT", "60"))

# ---------------------------------------------------------------------------
# Dataclass Topic (runtime)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Topic:
    name: str
    keywords: List[str]
    lemma_set: Set[str]
    fuzzy_norm: str
    vector: np.ndarray  # Ritorniamo a usare np.ndarray

# ---------------------------------------------------------------------------
# Helper build per metadati offline (usato già da QuestionImporter) ---------
# ---------------------------------------------------------------------------

class TopicMetaBuilder:
    """Costruisce *lemma_set*, *fuzzy_norm* e *vector* per un sub‑topic."""

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", unidecode.unidecode(text.lower().strip()))

    @classmethod
    def build(cls, keywords: List[str]) -> Tuple[List[str], str, List[float]]:
        """Restituisce (lemma_set, fuzzy_norm, vector_norm)."""
        lemmas = cls._get_lemmas(keywords)
        norm_string = cls._normalise(" ".join(keywords))
        # Sempre uso il servizio NLP per il vettore (richiede sentence-transformers)
        doc_vec = cls._get_vector(norm_string)
        # Normalizzazione locale con numpy
        vec = cls._normalize_vector(doc_vec)
        return list(lemmas), norm_string, vec.tolist()  # Converto in lista per salvare in JSON
        
    @staticmethod
    def _get_lemmas(keywords: List[str]) -> Set[str]:
        """Ottiene i lemmi dal servizio NLP."""
        try:
            response = requests.post(NLP_URL, json={"text": " ".join(keywords)})
            response.raise_for_status()
            data = response.json()
            lemmas = {token["lemma"] for token in data["tokens"]}
            return lemmas
        except Exception as e:
            logging.error(f"Errore nella chiamata al servizio NLP per lemmi: {e}")
            return set()  # Fallback a set vuoto in caso di errore
    
    @staticmethod
    def _get_vector(text: str) -> np.ndarray:
        """Ottiene il vettore dal servizio NLP."""
        try:
            response = requests.post(NLP_URL, json={"text": text})
            response.raise_for_status()
            data = response.json()
            vector = np.array(data["vector"])
            return vector
        except Exception as e:
            logging.error(f"Errore nella chiamata al servizio NLP per vettore: {e}")
            return np.zeros(300)  # Fallback a vettore di zeri in caso di errore

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        """Normalizza un vettore usando numpy localmente."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector  # Ritorna il vettore non normalizzato se la norma è 0

# ---------------------------------------------------------------------------
# Runtime utilities ---------------------------------------------------------
# Alias per retro‑compatibilità con interview_state (topic_from_meta) ---------
# ---------------------------------------------------------------------------

def topic_from_meta(
    subtopics: List[str],
    lemma_sets: List[List[str]],
    fuzzy_norms: List[str],
    vectors: List[List[float]],
    keywords: Optional[List[List[str]]] = None,
) -> List[Topic]:
    """Wrapper retro‑compatibile; forwards to topic_objects_from_meta.

    Se *keywords* è None si assume che queste non servano per la logica di
    coverage ma manteniamo la stessa firma del nuovo helper.
    """
    if keywords is None:
        keywords = [[] for _ in subtopics]
    return topic_objects_from_meta(subtopics, keywords, lemma_sets, fuzzy_norms, vectors)

# ---------------------------------------------------------------------------
# Runtime utilities ---------------------------------------------------------
# ---------------------------------------------------------------------------

def topic_objects_from_meta(
    subtopics: List[str],
    keywords: List[List[str]],
    lemma_sets: List[List[str]],
    fuzzy_norms: List[str],
    vectors: List[List[float]],
) -> List[Topic]:
    """Trasforma liste parallele (come in DOMANDE) in oggetti Topic."""
    topics: List[Topic] = []
    for name, kw, lem, fn, vec in zip(subtopics, keywords, lemma_sets, fuzzy_norms, vectors):
        topics.append(
            Topic(
                name=name,
                keywords=kw,
                lemma_set=set(lem),
                fuzzy_norm=fn,
                vector=np.asarray(vec, dtype=np.float32),  # Converti la lista in np.ndarray
            )
        )
    return topics


def _norm_user(text: str) -> Tuple[str, Set[str]]:
    """Ritorna (text_normalised, lemmi_set)."""
    text_norm = re.sub(r"\s+", " ", unidecode.unidecode(text.lower()))
    lemmi = TopicMetaBuilder._get_lemmas([text_norm])
    return text_norm, lemmi


def detect_covered_topics(user_text: str, topics: List[Topic]) -> Tuple[Set[str], float]:
    """Ritorna (set subtopic coperti, coverage_fraction 0‑1)."""
    if not user_text.strip():  # nessuna risposta
        return set(), 0.0

    txt_norm, user_lemmi = _norm_user(user_text)
    if token_sort_ratio is None:
        raise ImportError("rapidfuzz non installato – richiesto per fuzzy matching")

    # vettore utente solo se servirà il livello 3
    user_vec = None

    remaining = set(t.name for t in topics)
    covered: Set[str] = set()

    # ---- Livello 1: exact lemma --------------------------------------
    for t in topics:
        if t.name in remaining and t.lemma_set.intersection(user_lemmi):
            covered.add(t.name)
            remaining.remove(t.name)

    # ---- Livello 2: fuzzy -------------------------------------------
    for t in topics:
        if t.name not in remaining:
            continue
        score = token_sort_ratio(txt_norm, t.fuzzy_norm)
        if score >= TH_FUZZY:
            covered.add(t.name)
            remaining.remove(t.name)

    # ---- Livello 3: cosine ------------------------------------------
    if remaining:
        if user_vec is None:
            doc_vec = TopicMetaBuilder._get_vector(txt_norm)
            # Normalizzazione locale con numpy
            user_vec = TopicMetaBuilder._normalize_vector(doc_vec)
        for t in topics:
            if t.name not in remaining:
                continue
            # Calcolo locale della similarità coseno con numpy
            cos = _calculate_cosine_similarity(user_vec, t.vector)
            if cos >= TH_COS:
                covered.add(t.name)
                remaining.remove(t.name)

    coverage = 1 - len(remaining) / len(topics) if topics else 0.0
    return covered, coverage


def _calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calcola la similarità coseno usando numpy localmente."""
    return float(np.dot(vec1, vec2))

# ---------------------------------------------------------------------------
# Fine modulo
# ---------------------------------------------------------------------------
