"""
Adattatore per la classe InterviewState esterna.

Questo modulo fornisce un'interfaccia tra la classe InterviewState definita
nel file esterno interview_state.py e il sistema interno di gestione delle interviste.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json
import logging
import asyncio
import uuid
from Main.core.logger import logger
import Main.core.config as config
import tempfile
from typing import Union, List, Dict, Tuple, Any, Optional

# Import evitando cicli di importazione circolari
# Definizione di LocalInterviewState usato in questo file
class LocalInterviewState:
    """Classe locale per la gestione dello stato dell'intervista, definita qui per evitare import circolari"""
    def __init__(self, user_id, session_id=None, questions=None):
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())
        self.questions = questions or []
        self.current_question_id = None
        self.questions_asked = []
        self.answers = {}

from .user_session_service import DOMANDE, SCRIPT
try:
    from interview_state import InterviewState as ExternalInterviewState
    logger.info("Importata implementazione esterna di InterviewState con funzionalità di missing_topics")
except ImportError:
    logger.warning("Non è stato possibile importare la classe esterna InterviewState")
    ExternalInterviewState = None


# Verifica se DEVELOPMENT_MODE è disponibile, altrimenti usa un valore predefinito
try:
    from Main.core.config import DEVELOPMENT_MODE
except ImportError:
    DEVELOPMENT_MODE = False

# Funzioni di persistenza - stub se non importabili
try:
    from Main.services.persistence_service import save_interview_question, save_interview_response
except ImportError:
    # Se non possiamo importare le funzioni reali, creiamo degli stub
    def save_interview_question(*args, **kwargs):
        logger.warning("Funzione save_interview_question non disponibile - nessun dato salvato")
        return True
        
    def save_interview_response(*args, **kwargs):
        logger.warning("Funzione save_interview_response non disponibile - nessun dato salvato")
        return True
        
# Importazione condizionale del servizio LLM
try:
    from Main.services.llm_service import _follow_up_async
except ImportError:
    # Se non possiamo importare la funzione reale, creiamo uno stub
    async def _follow_up_async(*args, **kwargs):
        logger.warning("Funzione _follow_up_async non disponibile - uso fallback")
        return "Potresti fornire più dettagli su questo argomento?"

# Configurazione logger
logger = logging.getLogger(__name__)

# Dizionario globale delle sessioni - Usiamo LocalInterviewState come tipo
SESSIONS: Dict[str, LocalInterviewState] = {}

# Script globale caricato (le domande per l'intervista)
SCRIPT: List[Dict[str, Any]] = []


def convert_to_internal_question_format(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converte le domande dal formato esterno al formato interno utilizzato dall'API.
    
    Args:
        questions: Lista di domande nel formato esterno
        
    Returns:
        Lista di domande nel formato interno
    """
    result = []
    for i, q in enumerate(questions):
        question_id = q.get("id", f"q{i+1}")
        internal_question = {
            "id": question_id,
            "text": q.get("question", q.get("text", "")),
            "category": q.get("category", "general"),
            "difficulty": q.get("difficulty", "medium")
        }
        
        # Aggiungi topic e subtopics se disponibili
        if "topics" in q:
            internal_question["topic"] = q["topics"][0] if q["topics"] else None
            internal_question["subtopics"] = q["topics"][1:] if len(q["topics"]) > 1 else []
            
        result.append(internal_question)
        
    return result


def get_state(uid: str) -> LocalInterviewState:
    """
    Recupera lo stato dell'intervista per un utente o ne crea uno nuovo.
    
    Args:
        uid: ID dell'utente
        
    Returns:
        Istanza di InterviewState
    """
    if uid not in SESSIONS:
        # Usa lo script globale o un placeholder se vuoto
        current_script = SCRIPT
        if not current_script:
            logger.warning(f"SCRIPT globale è vuoto per user_id '{uid}'. Uso un set di domande placeholder.")
            current_script = [
                {"id": "q1", "text": "Parlami di te.", "topics": ["background", "introduction"], "category": "background", "difficulty": "medium"},
                {"id": "q2", "text": "Quali sono i tuoi obiettivi professionali?", "topics": ["goals", "career"], "category": "goals", "difficulty": "easy"},
                {"id": "q3", "text": "Descrivi una sfida che hai affrontato in ambito lavorativo.", "topics": ["experience", "challenges"], "category": "experience", "difficulty": "hard"}
            ]
        
        # Creiamo un nuovo stato usando l'implementazione locale
        # Passiamo correttamente i parametri: user_id, session_id=None (per generare UUID), questions=current_script
        SESSIONS[uid] = LocalInterviewState(uid, None, current_script)
        
        # Per inizializzare lo stato, facciamo avanzare alla prima domanda
        # Dobbiamo assicurarci che la prima domanda sia già impostata come domanda corrente
        # per evitare la ripetizione della prima domanda
        if hasattr(SESSIONS[uid], 'get_next_question') and callable(SESSIONS[uid].get_next_question):
            try:
                next_q = SESSIONS[uid].get_next_question()
                logger.debug(f"Prima domanda inizializzata: {next_q}")
                
                # Impostiamo la domanda come corrente per evitare la ripetizione
                if next_q and 'id' in next_q:
                    SESSIONS[uid].current_question_id = next_q['id']
                    logger.info(f"Domanda corrente inizializzata con ID: {next_q['id']}")
                    
                    # Aggiungiamo alla lista delle domande già poste per assicurarci che non venga ripetuta
                    if hasattr(SESSIONS[uid], 'questions_asked'):
                        if next_q['id'] not in SESSIONS[uid].questions_asked:
                            SESSIONS[uid].questions_asked.append(next_q['id'])
                            logger.debug(f"Aggiunto ID {next_q['id']} alla lista delle domande già poste")
            except Exception as e:
                logger.error(f"Errore durante l'inizializzazione della prima domanda: {e}")
    
    return SESSIONS[uid]


def has_active_session(uid: str) -> bool:
    """Verifica se esiste una sessione attiva per l'utente."""
    return uid in SESSIONS


def reset_session(uid: str) -> bool:
    """Elimina una sessione utente se esiste."""
    if uid in SESSIONS:
        del SESSIONS[uid]
        logger.info(f"Sessione eliminata per l'utente {uid}")
        return True
    return False


def load_script(questions: List[Dict[str, Any]]) -> bool:
    """
    Carica un nuovo script di domande.
    
    Args:
        questions: Lista di domande nel formato interno o esterno
        
    Returns:
        True se il caricamento è avvenuto con successo
    """
    global SCRIPT
    try:
        SCRIPT = questions
        logger.info(f"Nuovo script caricato con {len(SCRIPT)} domande")
        return True
    except Exception as e:
        logger.error(f"Errore nel caricamento dello script: {e}")
        return False


def get_current_question(state) -> Dict[str, Any]:
    """
    Ottiene la domanda corrente dalla sessione.
    Gestisce sia le domande principali che le domande di follow-up.
    
    Args:
        state: Istanza della classe InterviewState
        
    Returns:
        Dizionario con i dati della domanda corrente
    """
    if not state:
        logger.error("Stato dell'intervista non valido o mancante")
        return {
            "id": "error",
            "text": "Errore nel recupero della domanda",
            "category": "error",
            "difficulty": "medium"
        }
    
    # Verifica se l'oggetto è una InterviewState interna
    is_internal_state = hasattr(state, 'questions') and hasattr(state, 'current_question_id')
    
    logger.debug(f"Tipo di stato: {'Interno' if is_internal_state else 'Sconosciuto'}")
    
    # Impostazione iniziale dell'ID della domanda
    question_id = "q1"  # Default sicuro
    if is_internal_state:
        question_id = state.current_question_id or "q1"
        
    result = None
    
    try:
        # Verifica prima se abbiamo una domanda di follow-up in corso
        has_follow_up = hasattr(state, 'current_question_is_follow_up_for_subtopic') and state.current_question_is_follow_up_for_subtopic is not None
        
        if has_follow_up:
            logger.info(f"Recupero domanda di follow-up per subtopic: {state.current_question_is_follow_up_for_subtopic}")
            
            # Cerca il follow-up question nel campo della sessione o in altre strutture
            # Poiché le domande di follow-up sono generate dinamicamente e non fanno parte dello script originale
            result = {
                "id": f"{question_id}_followup",
                "text": None,  # Inizializziamo a None e lo popoleremo dopo
                "type": "follow_up",
                "category": "follow_up",
                "difficulty": "medium",
                "follow_up_for": state.current_question_is_follow_up_for_subtopic
            }
            
            # Ottieni il testo effettivo della domanda di follow-up
            try:
                # Nel caso normale, il testo della domanda sarebbe memorizzato nella sessione
                # dopo la generazione in save_answer()
                # Se la domanda di follow-up non è memorizzata, usa una fallback
                if hasattr(state, 'follow_up_question') and state.follow_up_question:
                    result["text"] = state.follow_up_question
                    logger.debug(f"Recuperata domanda di follow-up dalla sessione: {result['text']}")
                else:
                    # Se non troviamo la domanda salvata, generiamo una generica
                    missing_topic = state.current_question_is_follow_up_for_subtopic
                    fallback_text = f"Potresti dirmi di più riguardo a '{missing_topic}'?"
                    result["text"] = fallback_text
                    logger.warning(f"Generata domanda di follow-up fallback: {fallback_text}")
            except Exception as e:
                logger.error(f"Errore nel recupero del testo della domanda di follow-up: {e}")
                result["text"] = f"Potresti dirmi qualcosa di più sull'argomento?"
        elif is_internal_state:
            # Caso stato interno: usa la domanda corrente dall'oggetto InterviewState interno
            logger.debug(f"Recupero domanda principale per domanda ID {state.current_question_id}")
            
            # Trova la domanda corrente nell'elenco delle domande
            current_question = None
            for q in state.questions:
                if q.get('id') == state.current_question_id:
                    current_question = q
                    break
            
            if current_question:
                # Usa la domanda trovata
                result = {
                    "id": current_question.get('id', question_id),
                    "text": current_question.get('text', current_question.get('Domanda', 'Domanda non disponibile')),
                    "category": current_question.get('category', 'general'),
                    "difficulty": current_question.get('difficulty', 'medium'),
                    "type": "main"
                }
            else:
                # Se non abbiamo una domanda corrente, restituisci la prima disponibile
                if state.questions:
                    first_q = state.questions[0]
                    result = {
                        "id": first_q.get('id', 'q1'),
                        "text": first_q.get('text', first_q.get('Domanda', 'Prima domanda')),
                        "category": first_q.get('category', 'general'),
                        "difficulty": first_q.get('difficulty', 'medium'),
                        "type": "main"
                    }
                    # Aggiorna la domanda corrente nello stato
                    state.current_question_id = result['id']
                else:
                    # Non ci sono domande disponibili
                    result = {
                        "id": "no_questions",
                        "text": "Non ci sono domande disponibili.",
                        "category": "error",
                        "difficulty": "medium",
                        "type": "main"
                    }
        
        # Log della domanda recuperata per debug
        logger.debug(f"Domanda recuperata - ID: {result['id']}, Tipo: {result.get('type', 'main')}, Testo: {result['text'][:50]}...")
        return result
    except Exception as e:
        logger.error(f"Errore nell'ottenere la domanda corrente: {e}", exc_info=True)
        return {
            "id": "error",
            "text": "Errore nel recupero della domanda",
            "category": "error",
            "difficulty": "medium"
        }


def save_answer(state, user_response: str) -> Tuple[bool, float, List[str]]:
    """
    Salva la risposta dell'utente e verifica se è necessario un follow-up.
    
    Args:
        state: Istanza della classe InterviewState
        user_response: Risposta testuale dell'utente
        
    Returns:
        Tuple[bool, float, List[str]]: 
            - Flag che indica se è necessario un follow-up
            - Percentuale di copertura dei sottotopici
            - Lista dei topic mancanti
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*80)
    logger.info(f"SALVATAGGIO RISPOSTA UTENTE: {user_response[:200]}...")
    logger.info("="*80)
    
    # Log dello stato corrente
    logger.debug(f"Stato corrente - IDX: {getattr(state, 'idx', 'N/A')}")
    logger.debug(f"Domanda corrente: {getattr(state, 'current_question', 'Nessuna domanda corrente')}")
    logger.debug(f"Topic corrente: {getattr(state, 'current_topic', 'Nessun topic')}")
    logger.debug(f"Subtopics: {getattr(state, 'current_subtopics', [])}")
    logger.debug(f"Keywords: {getattr(state, 'current_keywords', {})}")

    if not state:
        logger.error("Stato dell'intervista non valido o mancante")
        return False, 0.0, []
    
    # Verifica se l'oggetto è InterviewState
    is_internal_state = hasattr(state, 'questions') and hasattr(state, 'current_question_id')
    
    logger.debug(f"Save answer - Tipo di stato: {'Interno' if is_internal_state else 'Sconosciuto'}")
    
    try:
        # Se abbiamo lo stato interno standard
        if is_internal_state:
            # Otteniamo l'ID della domanda corrente
            current_id = state.current_question_id
            
            # Verifica che current_question_id non sia None
            if current_id is None and hasattr(state, 'questions') and len(state.questions) > 0:
                # Se è None, inizializza con il primo ID disponibile
                state.current_question_id = state.questions[0].get('id')
                logger.info(f"Inizializzato current_question_id con il primo ID disponibile: {state.current_question_id}")
            
            # Se non abbiamo già un dizionario di risposte, lo creiamo
            if not hasattr(state, 'answers') or state.answers is None:
                state.answers = {}
                
            # Salviamo la risposta
            state.answers[current_id] = user_response
            logger.debug(f"Salvata risposta per domanda {current_id}")
            
            # Aggiorna i metadati della domanda corrente
            if current_id:
                # Recupera i metadati della domanda corrente
                from .user_session_service import get_question_metadata_status
                
                try:
                    # Recupero metadati con gestione errori
                    metadata = get_question_metadata_status(current_id)
                    
                    if metadata and metadata['status'] == 'completed' and metadata.get('metadata'):
                        metadata_content = metadata['metadata']
                        logger.info(f"Aggiornamento metadati per domanda {current_id}")
                        
                        # 1. Metadati base (topic e subtopics)
                        primary_topic = metadata_content.get('primary_topic')
                        subtopics = metadata_content.get('subtopics', [])
                        
                        # Aggiornamento attraverso setter se disponibili, altrimenti direttamente
                        if hasattr(state, 'set_current_topic'):
                            state.set_current_topic(primary_topic)
                        else:
                            state.current_topic = primary_topic
                            
                        if hasattr(state, 'set_current_subtopics'):
                            state.set_current_subtopics(subtopics)
                        else:
                            state.current_subtopics = subtopics
                        
                        # 2. Metadati avanzati (keywords e strutture correlate)
                        keywords = metadata_content.get('keywords', [])
                        if hasattr(state, 'set_current_keywords'):
                            state.set_current_keywords(keywords)
                        else:
                            state.current_keywords = keywords
                        
                        # 3. Metadati per analisi semantica dal microservizio
                        if 'vectors' in metadata_content:
                            vectors = metadata_content.get('vectors', [])
                            if hasattr(state, 'set_current_vectors'):
                                state.set_current_vectors(vectors)
                            else:
                                state.current_vectors = vectors
                            logger.debug(f"Aggiornati {len(vectors)} vettori per analisi semantica")
                        
                        if 'lemma_sets' in metadata_content:
                            lemma_sets = metadata_content.get('lemma_sets', [])
                            if hasattr(state, 'set_current_lemma_sets'):
                                state.set_current_lemma_sets(lemma_sets)
                            else:
                                state.current_lemma_sets = lemma_sets
                            logger.debug(f"Aggiornati {len(lemma_sets)} lemma sets")
                        
                        if 'fuzzy_norms' in metadata_content:
                            fuzzy_norms = metadata_content.get('fuzzy_norms', [])
                            if hasattr(state, 'set_current_fuzzy_norms'):
                                state.set_current_fuzzy_norms(fuzzy_norms)
                            else:
                                state.current_fuzzy_norms = fuzzy_norms
                            logger.debug(f"Aggiornati {len(fuzzy_norms)} fuzzy norms")
                        
                        # 4. Cache temporanea per ottimizzare le prestazioni future
                        # Salviamo i metadati completi in una cache locale nello stato
                        if not hasattr(state, 'metadata_cache'):
                            state.metadata_cache = {}
                        
                        # Memorizza i metadati completi nella cache (evita chiamate ripetute)
                        state.metadata_cache[current_id] = metadata_content
                        
                        logger.info(f"Metadati aggiornati con successo: Topic={state.current_topic}, Subtopics={state.current_subtopics}")
                        # Salva la risposta nel database con tutti i metadati
                        try:
                            # Ottieni la domanda corrente
                            current_question = None
                            for q in state.questions:
                                if q.get('id') == current_id:
                                    current_question = q
                                    break
                            
                            if current_question:
                                # Prepara i metadati per il salvataggio
                                question_idx = state.questions.index(current_question)
                                question_text = current_question.get('text', current_question.get('Domanda', ''))
                            
                            # Salva la risposta con tutti i metadati disponibili
                            save_interview_response(
                                user_id=state.user_id,
                                session_id=state.session_id,
                                question_idx=question_idx,
                                question_text=question_text,
                                response_text=user_response,
                                topic=state.current_topic,
                                subtopics=state.current_subtopics,
                                keywords=getattr(state, 'current_keywords', {}),
                                non_covered_subtopics=getattr(state, 'missing_subtopics', []),
                                coverage_percent=getattr(state, 'coverage_percent', 0.0)
                            )
                            logger.info(f"Risposta salvata nel database con metadati completi per domanda {current_id}")
                        except Exception as e:
                            logger.error(f"Errore durante il salvataggio della risposta nel database: {e}")
                            # Continuiamo l'esecuzione anche in caso di errore
                except Exception as e:
                    logger.error(f"Errore durante l'aggiornamento dei metadati: {e}", exc_info=True)
                    # Continuiamo l'esecuzione anche in caso di errore
            
            # Se lo stato ha il metodo add_answer, lo utilizziamo
            if hasattr(state, 'add_answer') and callable(state.add_answer):
                try:
                    state.add_answer(current_id, user_response)
                    logger.debug(f"Risposta salvata anche tramite add_answer per {current_id}")
                except Exception as e:
                    logger.error(f"Errore durante l'esecuzione di add_answer: {e}")
        
        # Ottieni gli argomenti mancanti
        # Analizziamo la risposta per determinare se è necessario un follow-up
        missing_topics = []
        coverage_percent = 100.0  # Default: considera tutto coperto
        
        # Soglia di copertura per decidere se è necessario un follow-up
        COVERAGE_THRESHOLD_PERCENT = 80.0
        
        # Controlliamo se il metodo missing_topics è disponibile
        if not hasattr(state, 'missing_topics') or not callable(state.missing_topics):
            logger.info("Metodo missing_topics non presente nello stato - verifico se posso usare l'implementazione esterna")
            
            # Verifichiamo se esiste l'implementazione esterna di InterviewState
            if ExternalInterviewState is not None:
                logger.info("Adattamento della sessione per utilizzare l'implementazione esterna di missing_topics")
                # Controlliamo se l'oggetto state ha gli attributi necessari per missing_topics
                
                # Aggiungiamo gli attributi necessari all'implementazione esterna
                if not hasattr(state, 'script') and hasattr(state, 'questions'):
                    state.script = state.questions
                if not hasattr(state, 'current_question_is_follow_up_for_subtopic'):
                    state.current_question_is_follow_up_for_subtopic = None
                    logger.debug("Inizializzato current_question_is_follow_up_for_subtopic a None")
                
                if not hasattr(state, 'idx') and hasattr(state, 'current_question_id'):
                    # Verifica che current_question_id non sia None
                    if state.current_question_id is None and hasattr(state, 'questions') and len(state.questions) > 0:
                        # Se è None, inizializza con il primo ID disponibile
                        state.current_question_id = state.questions[0].get('id')
                        logger.info(f"Inizializzato current_question_id con il primo ID disponibile: {state.current_question_id}")
                    
                    # Troviamo l'indice della domanda corrente
                    for i, q in enumerate(state.questions):
                        if q.get('id') == state.current_question_id:
                            state.idx = i
                            break
                    else:
                        state.idx = 0
                
                # Aggiungiamo un oggetto transcript simulato se necessario
                if not hasattr(state, 'rm'):
                    class MockResponseManager:
                        def __init__(self):
                            self.transcript = []
                            
                        def add_user_response(self, text):
                            self.transcript.append({"speaker": "user", "text": text})
                    
                    state.rm = MockResponseManager()
                    # Aggiungiamo la risposta corrente al transcript
                    state.rm.add_user_response(user_response)
                
                # Prendiamo il metodo dalla classe esterna e lo leghiamo al nostro oggetto state
                try:
                    # Creiamo una copia del metodo missing_topics dall'implementazione esterna
                    state.missing_topics = ExternalInterviewState.missing_topics.__get__(state, type(state))
                    # Aggiungiamo anche il metodo domanda_corrente
                    state.domanda_corrente = ExternalInterviewState.domanda_corrente.__get__(state, type(state))
                    logger.info("Metodi missing_topics e domanda_corrente aggiunti con successo dallo stato esterno")
                except Exception as e:
                    logger.error(f"Errore nell'aggiungere i metodi dallo stato esterno: {e}")
                    # Fallback semplice se l'integrazione non funziona
                    state.missing_topics = lambda: ([], 100.0)
                    # Fallback per domanda_corrente: restituisce la domanda corrente dal dizionario questions
                    state.domanda_corrente = lambda: state.questions[state.idx].get("question", "Domanda non disponibile") if hasattr(state, "questions") and hasattr(state, "idx") and state.idx < len(state.questions) else "Domanda non disponibile"
            else:
                logger.warning("Implementazione esterna non disponibile, utilizzando fallback semplice")
                # Fallback molto semplice che non richiede follow-up
                state.missing_topics = lambda: ([], 100.0)
                # Fallback per domanda_corrente: restituisce la domanda corrente dal dizionario questions
                state.domanda_corrente = lambda: state.questions[state.idx].get("question", "Domanda non disponibile") if hasattr(state, "questions") and hasattr(state, "idx") and state.idx < len(state.questions) else "Domanda non disponibile"
        
        # Ora usiamo il metodo missing_topics (originale o simulato)
        if hasattr(state, 'missing_topics') and callable(state.missing_topics):
            try:
                logger.info("\n" + "-"*40)
                logger.info("INIZIO ANALISI RISPOSTA UTENTE")
                logger.info("-"*40)
                
                missing_topics, coverage_percent = state.missing_topics()
                
                # Rimuoviamo eventuali duplicati dai sottotopici mancanti
                missing_topics = list(set(missing_topics)) if missing_topics else []
                
                # Log dettagliato dei topics e della copertura
                logger.info(f"\nRISULTATO ANALISI:")
                logger.info(f"- Topic mancanti: {missing_topics}")
                logger.info(f"- Percentuale copertura: {coverage_percent:.1f}% (soglia: {COVERAGE_THRESHOLD_PERCENT}%)")
                
                # Log dei topics correnti se disponibili
                current_topic = getattr(state, 'current_topic', 'Nessun topic')
                current_subtopics = getattr(state, 'current_subtopics', [])
                
                logger.info("\nCONTESTO ATTUALE:")
                logger.info(f"- Topic corrente: {current_topic}")
                logger.info(f"- Subtopics: {current_subtopics}")
                logger.info("-"*40 + "\n")
                
            except Exception as e:
                logger.error(f"ERRORE durante l'analisi dei topics: {e}", exc_info=True)
                missing_topics = []
                coverage_percent = 100.0  # Fallback: considera tutto coperto in caso di errore
        else:
            logger.warning("ATTENZIONE: Il metodo missing_topics non è disponibile nello stato corrente")
        
        # Decidiamo se è necessario un follow-up in base alla percentuale di copertura
        needs_followup = coverage_percent < COVERAGE_THRESHOLD_PERCENT and missing_topics
        
        if needs_followup and is_internal_state:
            try:
                # Funzionalità di follow-up disponibile solo per lo stato interno attualmente
                # Selezioniamo un sottotopico mancante per la domanda di follow-up
                # (semplice, prendiamo il primo)
                selected_subtopic = missing_topics[0]
                
                # Logica per generare una domanda di follow-up
                logger.info(f"Generazione domanda di follow-up per sottotopico: {selected_subtopic}")
                
                # Otteniamo il testo della domanda corrente
                try:
                    current_question = state.domanda_corrente()
                except Exception as e:
                    logger.error(f"Errore durante l'accesso alla domanda corrente: {e}")
                    current_question = "Domanda non disponibile"
                follow_up_text = None
                
                # Otteniamo la domanda di follow-up attraverso la funzione asincrona
                # in modo sincrono (blocca fino a completamento)
                try:
                    logger.debug(f"Current question: {current_question}")
                    logger.debug(f"User response: {user_response}")
                    
                    # Fallback per get_notes() se non disponibile
                    if hasattr(state, 'get_notes') and callable(getattr(state, 'get_notes')):
                        reflection = state.get_notes() or "Nessuna nota disponibile"
                    else:
                        # Crea una riflessione di base basata sulla domanda e risposta
                        reflection = f"Domanda: {current_question}\nRisposta: {user_response}"
                        logger.warning("Metodo get_notes() non disponibile, usando riflessione di base")
                    
                    logger.debug(f"Reflection: {reflection}")
                    logger.debug(f"Missing topics: {missing_topics}")
                    
                    follow_up_task = asyncio.create_task(
                        _follow_up_async(
                            current_question, 
                            user_response, 
                            reflection, 
                            missing_topics
                        )
                    )
                    # Attendiamo il risultato con un timeout
                    follow_up_text = asyncio.get_event_loop().run_until_complete(
                        asyncio.wait_for(follow_up_task, timeout=15.0)  # Aumentato a 15 secondi
                    )
                    logger.debug(f"Domanda di follow-up generata con successo: {follow_up_text}")
                except asyncio.TimeoutError:
                    logger.warning("Timeout nella generazione della domanda di follow-up (tentativo 1)")
                except Exception as e:
                    logger.error(f"Errore nel generare la domanda di follow-up (tentativo 1): {e}")
                
                # Se il primo tentativo fallisce, proviamo con un fallback più semplice
                if not follow_up_text:
                    try:
                        logger.info("Tentativo alternativo di generazione domanda follow-up...")
                        if selected_subtopic:
                            follow_up_text = f"Potresti dirmi di più riguardo a '{selected_subtopic}'?"
                            logger.info(f"Domanda di follow-up di fallback generata: {follow_up_text}")
                        else:
                            follow_up_text = "Puoi approfondire meglio questo argomento?"
                            logger.info("Nessun subtopic selezionato, usando domanda generica di follow-up")
                    except Exception as e2:
                        logger.error(f"Errore anche nel fallback per la domanda di follow-up: {e2}")
                        follow_up_text = "Puoi approfondire meglio questo argomento?"
                
                # Se abbiamo una domanda di follow-up, la impostiamo nello stato
                if follow_up_text:
                    logger.info(f"Domanda follow-up generata: {follow_up_text}")
                    state.follow_up_question = follow_up_text
                    state.current_question_is_follow_up_for_subtopic = selected_subtopic
                    
                    # Non avanziamo alla prossima domanda in questo caso,
                    # poiché faremo prima la domanda di follow-up
                    return needs_followup, coverage_percent, missing_topics
                
                logger.error("Impossibile generare domanda di follow-up, procedo con la domanda successiva")
                # In caso di fallimento, procediamo con la domanda successiva
                advance_to_next_question(state)
                
            except Exception as e:
                logger.error(f"Errore nel processo di generazione domanda follow-up: {e}")
                # In caso di errore, procediamo con la domanda successiva
                advance_to_next_question(state)
        else:
            # Non è necessario un follow-up, procediamo con la prossima domanda
            logger.info("Nessun follow-up necessario, procedo con la prossima domanda")
            advance_to_next_question(state)
            
            # Se c'è una domanda di follow-up, assicuriamoci che needed_followup sia True
            if hasattr(state, 'follow_up_question') and state.follow_up_question:
                needs_followup = True
        
        # Salva la risposta utilizzando il servizio di persistenza
        try:
            # Usa l'indice corretto in base all'implementazione di InterviewState
            current_q_idx = getattr(state, 'idx', 0)  # Usa 0 come fallback se idx non esiste
            current_q_text = getattr(state, 'current_question', current_question if 'current_question' in locals() else "")
            
            # Log dei dettagli per debug
            logger.debug(f"Salvataggio risposta - IDX: {current_q_idx}, Domanda: {current_q_text}")
            
            save_interview_response(
                user_id=state.user_id,
                session_id=getattr(state, 'session_id', 'unknown_session'),
                question_idx=current_q_idx,
                question_text=current_q_text,
                response_text=user_response,
                topic=getattr(state, 'current_topic', None),
                subtopics=getattr(state, 'current_subtopics', []),
                non_covered_subtopics=missing_topics,
                coverage_percent=coverage_percent
            )
        except Exception as e:
            logger.error(f"Errore durante il salvataggio della risposta: {e}")
        
        return needs_followup, coverage_percent, missing_topics
    
    except Exception as e:
        logger.error(f"Errore nel salvataggio della risposta: {e}")
        return False, 0.0, []


def advance_to_next_question(state) -> bool:
    """
    Avanza alla prossima domanda dell'intervista.
    
    Args:
        state: Istanza della classe InterviewState
        
    Returns:
        True se l'avanzamento è avvenuto con successo, False altrimenti
    """
    try:
        # Resetta lo stato di follow-up se presente
        if hasattr(state, 'current_question_is_follow_up_for_subtopic'):
            state.current_question_is_follow_up_for_subtopic = None
            
        if hasattr(state, 'follow_up_question'):
            state.follow_up_question = None
            
        # Per lo stato interno standard, passa alla domanda successiva
        if hasattr(state, 'questions') and hasattr(state, 'current_question_id'):
            # Ottiene l'ID della domanda corrente
            current_id = state.current_question_id
            current_idx = -1
            
            # Trova l'indice della domanda corrente
            for i, q in enumerate(state.questions):
                if q.get('id') == current_id:
                    current_idx = i
                    break
            
            # Se abbiamo trovato la domanda corrente e non siamo all'ultima domanda
            if current_idx >= 0 and current_idx < len(state.questions) - 1:
                # Avanza alla prossima domanda
                next_idx = current_idx + 1
                state.current_question_id = state.questions[next_idx].get('id')
                logger.debug(f"Avanzato alla domanda {state.current_question_id} (indice {next_idx})")
            else:
                # Siamo alla fine delle domande o la domanda corrente non è stata trovata
                if current_idx < 0:
                    logger.warning(f"La domanda corrente con ID {current_id} non è stata trovata nell'elenco. Inizializziamo con la prima domanda.")
                    # Se la domanda corrente non è trovata, inizializziamo con la prima domanda
                    if state.questions and len(state.questions) > 0:
                        state.current_question_id = state.questions[0].get('id')
                        logger.info(f"Inizializzato con la prima domanda: {state.current_question_id}")
                else:
                    logger.info("Raggiunta l'ultima domanda dell'intervista")
            
            # Se lo stato ha l'attributo get_next_question, lo utilizziamo
            if hasattr(state, 'get_next_question') and callable(state.get_next_question):
                try:
                    next_q = state.get_next_question()
                    logger.debug(f"Prossima domanda ottenuta tramite get_next_question: {next_q}")
                except Exception as e:
                    logger.error(f"Errore durante l'esecuzione di get_next_question: {e}")
        
        # Tentativo di salvare la domanda corrente se ci sono gli attributi necessari
        if hasattr(state, 'user_id') and hasattr(state, 'interview_id'):
            try:
                # Ottieni la domanda corrente
                current_question = None
                for q in state.questions:
                    if q.get('id') == state.current_question_id:
                        current_question = q
                        break
                
                if current_question:
                    save_interview_question(
                        user_id=state.user_id,
                        session_id=state.interview_id,  # Usa interview_id come session_id
                        question_idx=state.questions.index(current_question),
                        question_text=current_question.get('text', current_question.get('Domanda', '')),
                        is_follow_up=False
                    )
            except Exception as e:
                logger.error(f"Errore durante il salvataggio della domanda: {e}")
            
        return True
        
    except Exception as e:
        logger.error(f"Errore nell'avanzamento alla prossima domanda: {e}")
        return False


def get_session_info() -> Dict[str, Any]:
    """
    Restituisce informazioni sulle sessioni attive.
    
    Returns:
        Dizionario con statistiche sulle sessioni
    """
    return {
        "active_sessions": len(SESSIONS),
        "session_ids": list(SESSIONS.keys()),
        "script_size": len(SCRIPT)
    }
