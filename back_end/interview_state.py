# interview_state.py
from typing import List, Dict, Union, Tuple, Optional, Any  # FIX: added Any for type hints
from interviewer_reflection import InterviewerReflection
import uuid
import logging
import os
from topic_detection import (
    COVERAGE_THRESHOLD_PERCENT as TD_COVERAGE_THRESHOLD_PERCENT,
    detect_covered_topics,
    topic_objects_from_meta,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costante di soglia: se presente nel .env sostituisce quella importata da
# topic_detection. Mantenere lo stesso *nome di variabile* usato altrove per
# non rompere le referenze.
# ---------------------------------------------------------------------------
COVERAGE_THRESHOLD_PERCENT = float(os.getenv("COVERAGE_THRESHOLD_PERCENT", TD_COVERAGE_THRESHOLD_PERCENT))


class InterviewState:
    """Gestisce lo stato di una singola intervista (RAM, nessun DB)."""

    def __init__(self, user_id: str, script: List[Dict[str, Union[str, List[str]]]]) -> None:
        self.user_id: str = user_id
        self.session_id: str = str(uuid.uuid4())  # mantiene compatibilità con DB
        self.script: List[Dict[str, Union[str, List[str]]]] = script
        self.idx: int = 0  # domanda corrente
        self.rm: InterviewerReflection = InterviewerReflection()  # gestore riflessioni

        # Nuovi campi per la struttura della domanda corrente generata dall'LLM
        self.current_topic: Optional[str] = None
        self.current_subtopics: List[str] = []
        self.current_keywords: List[List[str]] = []  # keywords per ogni subtopic
        self.current_question_is_follow_up_for_subtopic: Optional[str] = None

        # Lista per memorizzare le risposte dell'utente e i relativi metadati
        self.user_responses: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------------
    # Proprietà/alias per retro‑compatibilità con parti di codice legacy
    # ---------------------------------------------------------------------
    @property
    def question(self) -> str:  # alias usato da altri componenti che fanno obj.question
        """Restituisce la domanda corrente mantenendo compatibilità legacy."""
        return self.domanda_corrente()

    def get_notes(self) -> str:
        """Delegato che restituisce eventuali note dal riflettore.

        Alcuni moduli chiamano ``InterviewState.get_notes()`` per contestualizzare
        il prompt del LLM. Se ``InterviewerReflection`` espone già `get_notes`,
        lo usiamo; altrimenti costruiamo un fallback concatenando i turni
        dell'assistente.
        """
        if hasattr(self.rm, "get_notes") and callable(getattr(self.rm, "get_notes")):
            return self.rm.get_notes()
        # Fallback minimale
        return "\n".join(
            t.get("text", "") for t in self.rm.transcript if t.get("speaker") == "assistant"
        )

    # --------- API chiamate dal main.py ---------
    def domanda_corrente(self) -> str:
        """Restituisce il testo della domanda corrente."""
        try:
            # Ottieni la domanda da DOMANDE
            from Main.application.user_session_service import DOMANDE
            if DOMANDE and self.idx < len(DOMANDE):
                domanda = DOMANDE[self.idx]
                if isinstance(domanda, dict):
                    for key in ['question', 'text', 'domanda', 'testo']:
                        if key in domanda and domanda[key]:
                            return str(domanda[key])
        
        # Se arriviamo qui, non abbiamo trovato una domanda valida
            logger.warning(f"Domanda non trovata in DOMANDE all'indice {self.idx}")
            return "Domanda non disponibile"
        except Exception as e:
            logger.error(f"Errore nel recuperare la domanda corrente: {e}", exc_info=True)
            return "Domanda non disponibile"

    def save_user_response_and_reflect(self, text: str) -> None:
        """Salva risposta utente + riflettore."""
        self.rm.add_turn("user", text)

    def missing_topics(self) -> Tuple[List[str], float]:
        """Restituisce (subtopic mancanti, coverage_percent).

        Implementa la cascata exact‑lemma → fuzzy → cosine delegando al
        modulo ``topic_detection``.
        """
        # Importa DOMANDE
        from Main.application.user_session_service import DOMANDE

        # --------------- 0. Validità indice/script --------------------
        if not DOMANDE or self.idx >= len(DOMANDE):
            return [], 0.0

        q: Dict[str, Any] = DOMANDE[self.idx]
        expected_subtopics: List[str] = q.get("subtopics", [])
        if not expected_subtopics:
            return [], 100.0  # nessun topic definito

        # --------------- 1. Ricava meta per i sub-topic ---------------
        try:
            topics = topic_objects_from_meta(
                subtopics=q.get("subtopics", []),
                keywords=q.get("keywords", []),
                lemma_sets=q.get("lemma_sets", []),
                fuzzy_norms=q.get("fuzzy_norms", []),
                vectors=q.get("vectors", []),
            )

        except (KeyError, TypeError, ValueError) as e:
            # Fallback se lo YAML non è stato ancora arricchito o ha struttura errata
            logger.warning(
                "Metadata derivati mancanti o errati: %s – fallback keyword-only", e
            )
            subtopics = q.get("subtopics", [])
            keywords = q.get("keywords", [])

            # Verifica che keywords sia una lista di liste
            if not isinstance(keywords, list):
                keywords = []
            elif keywords and not isinstance(keywords[0], list):
                # Se keywords è una lista semplice, la trasformiamo in lista di liste
                keywords = [keywords]

            # Assicurati che keywords abbia la stessa lunghezza di subtopics
            while len(keywords) < len(subtopics):
                keywords.append([])

            topics = topic_objects_from_meta(
                subtopics=subtopics,
                keywords=keywords,
                lemma_sets=[[] for _ in subtopics],
                fuzzy_norms=["" for _ in subtopics],
                vectors=[[0.0] * 300 for _ in subtopics],  # spaCy dim.
            )

        # --------------- 2. Recupera l'ultima risposta utente ---------
        last_user_text: str = next(
            (
                t.get("text", "")
                for t in reversed(self.rm.transcript)
                if t.get("speaker") == "user"
            ),
            "",
        )
        if not last_user_text.strip():
            return expected_subtopics, 0.0  # nessuna risposta

        # --------------- 3. Applica cascata ---------------------------
        try:
            covered, coverage_frac = detect_covered_topics(last_user_text, topics)
            missing = [t.name for t in topics if t.name not in covered]
            coverage_percent = round(coverage_frac * 100, 1)
        except Exception as e:
            logger.error("Errore nell'analisi dei topic: %s", e)
            return expected_subtopics, 0.0  # fallback completo in caso di errore

        logger.debug(
            "Analisi risposta – coperti %s / %s (%.1f%%)",
            len(covered),
            len(topics),
            coverage_percent,
        )

        # --------------- 4. Decide follow‑up -------------------------
        if coverage_percent >= COVERAGE_THRESHOLD_PERCENT:
            return [], coverage_percent
        return missing, coverage_percent

    def advance_main(self) -> None:
        """Passa alla prossima domanda del copione."""
        self.idx = min(self.idx + 1, len(self.script) - 1)