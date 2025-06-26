from typing import Dict, List, Any, Union, Optional
import uuid
from datetime import datetime
import json
import logging
import tempfile
import time
import os
import threading

# Configurazione logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nuova struttura globale: DOMANDE = [q0, q1, q2, ...] dove ogni qi ha la struttura
# {"testo": "...", "topic": "...", "subtopics": [...], "keywords": {subtopic1: [...], ...}}
DOMANDE = []

# Manteniamo SCRIPT per retrocompatibilità, ma ora è secondario rispetto a DOMANDE
SCRIPT: List[Dict[str, Any]] = []

# Dizionario globale per tenere traccia dello stato di elaborazione delle domande
# Struttura: { domanda_id: { 'status': 'pending|processing|completed', 'metadata': {...} } }
METADATA_STATUS = {}

# Importiamo QuestionImporter per la generazione di metadati
try:
    from Importazioni import QuestionImporter
    logger.info("Importato QuestionImporter per la generazione di metadati")
except ImportError:
    logger.warning("Non è stato possibile importare QuestionImporter")
    QuestionImporter = None
    
# Stato di elaborazione metadati monitorabile dall'UI
metadata_processing_status = {
    'total_questions': 0,
    'processed_questions': 0,
    'in_progress': False,
    'start_time': None,
    'end_time': None,
    'error': None
}


# Classe per gestire lo stato dell'intervista
class InterviewState:
    """Gestisce lo stato dell'intervista e le domande/risposte"""
    def __init__(self, user_id: str, questions: List[Dict[str, Any]] = None):
        self.interview_id = str(uuid.uuid4())
        self.user_id = user_id
        self.start_time = datetime.now()
        self.current_question_id = None
        self.questions = questions or []  # Lista delle domande disponibili
        self.questions_asked = []  # ID delle domande già poste
        self.answers = {}  # Risposte fornite (question_id -> answer_text)
        self.completed = False
        self.score = None
        logger.info(f"Nuova sessione intervista creata: {self.interview_id} per utente: {user_id}")
    
    def get_next_question(self) -> Optional[Dict[str, Any]]:
        """Ottiene la prossima domanda da porre"""
        # Filtra le domande non ancora poste
        available_questions = [q for q in self.questions 
                             if q.get("id") not in self.questions_asked]
        
        if not available_questions:
            self.completed = True
            return None
            
        # Seleziona la prima domanda disponibile
        next_question = available_questions[0]
        question_id = next_question.get("id", str(len(self.questions_asked)))
        
        # Aggiorna lo stato dell'intervista
        self.current_question_id = question_id
        self.questions_asked.append(question_id)
        
        return next_question
    
    def add_answer(self, question_id: str, answer_text: str) -> bool:
        """Registra una risposta data dall'utente"""
        if question_id != self.current_question_id:
            logger.warning(f"Tentativo di rispondere a domanda non corrente: {question_id}")
            return False
            
        self.answers[question_id] = answer_text
        logger.info(f"Risposta registrata per domanda {question_id}")
        return True
    
    def complete_interview(self) -> int:
        """Termina l'intervista e calcola un punteggio"""
        self.completed = True
        
        # In una versione reale, calcoleremo un punteggio basato sulle risposte
        # Per ora, assegniamo un punteggio casuale
        import random
        self.score = random.randint(60, 95)
        
        logger.info(f"Intervista {self.interview_id} completata con punteggio: {self.score}")
        return self.score
        
# Dizionario globale delle sessioni 
SESSIONS: Dict[str, InterviewState] = {}

# Script globale caricato (le domande per l'intervista)
# Inizializziamo SCRIPT come una lista vuota - le domande verranno caricate dall'API
SCRIPT: List[Dict[str, Any]] = []  # Type hint per SCRIPT aggiornata

def get_state(uid: str) -> InterviewState:
    """Recupera la sessione di un utente o ne crea una nuova se non esiste"""
    if uid not in SESSIONS:
        current_script_for_session = SCRIPT
        if not current_script_for_session:
            logger.warning(f"SCRIPT globale è vuoto per user_id '{uid}'. Uso un set di domande placeholder.")
            current_script_for_session = [
                {"id": "q1", "text": "Parlami di te.", "category": "background", "difficulty": "medium"},
                {"id": "q2", "text": "Quali sono i tuoi obiettivi professionali?", "category": "goals", "difficulty": "easy"},
                {"id": "q3", "text": "Descrivi una sfida che hai affrontato in ambito lavorativo.", "category": "experience", "difficulty": "hard"}
            ]
        SESSIONS[uid] = InterviewState(uid, current_script_for_session)
        logger.info(f"Nuova sessione InterviewState creata per user_id '{uid}'.")
    return SESSIONS[uid]

def has_active_session(uid: str) -> bool:
    """Verifica se un utente ha una sessione attiva"""
    return uid in SESSIONS

def reset_session(uid: str) -> bool:
    """Elimina la sessione di un utente se esiste e restituisce True, altrimenti False"""
    if uid in SESSIONS:
        del SESSIONS[uid]
        logger.info(f"Sessione eliminata per user_id '{uid}'.")
        return True
    return False

def load_script(new_script: List[Dict[str, Any]]) -> bool:
    """
    Carica un nuovo script globale e avvia l'elaborazione asincrona dei metadati.
    Popola la struttura DOMANDE con le domande e i relativi metadati man mano che vengono elaborati.
    """
    global SCRIPT
    global DOMANDE  # Nuova struttura globale
    global metadata_processing_status
    
    try:
        # Svuotiamo la lista DOMANDE per il nuovo caricamento
        DOMANDE.clear()
        
        # Verifica che new_script non sia vuoto
        if not new_script:
            logger.error("Tentativo di caricare uno script vuoto! Operazione annullata.")
            return False
            
        # Verifica che gli elementi abbiano il formato corretto
        valid_items = []
        for i, item in enumerate(new_script):
            if not isinstance(item, dict):
                logger.error(f"Elemento {i} non è un dizionario: {type(item)}")
                continue
                
            if 'Domanda' not in item or not item['Domanda']:
                logger.error(f"Elemento {i} non ha una domanda valida: {item}")
                continue
            
            # Assicuriamoci che ogni domanda abbia un ID univoco
            if "id" not in item or not item["id"]:
                item["id"] = str(uuid.uuid4())
                
            valid_items.append(item)
            
        if not valid_items:
            logger.error("Nessuna domanda valida trovata nello script fornito!")
            return False
            
        # Assegna subito gli elementi validi per renderli disponibili immediatamente sia in SCRIPT che in DOMANDE
        SCRIPT = valid_items  # Manteniamo SCRIPT per retrocompatibilità
        
        # Inizializza DOMANDE con solo il testo delle domande (i metadati verranno aggiunti dopo)
        for i, question in enumerate(valid_items):
            domanda_testo = question.get('Domanda', '')
            q_struct = {
                "id": question.get("id"),
                "testo": domanda_testo,
                "topic": "",  # Sarà popolato durante l'elaborazione metadati
                "subtopics": [],  # Sarà popolato durante l'elaborazione metadati
                "keywords": [],   # Sarà popolato durante l'elaborazione metadati
                "lemma_sets": [],
                "fuzzy_norms": [],
                "vectors": []   
            }
            DOMANDE.append(q_struct)
        
        # Stampa dettagli per debug
        logger.info(f"Nuovo script caricato con {len(valid_items)} domande valide su {len(new_script)} fornite.")
        
        # Log delle domande caricate
        for i, question in enumerate(DOMANDE[:3]):  # Stampa solo le prime 3 come esempio
            logger.info(f"Domanda {i+1}: {question['testo'][:50]}...")
        
        # Generiamo i metadati DIRETTAMENTE usando QuestionImporter da Importazioni.py
        if QuestionImporter is not None:
            # Avvia l'elaborazione dei metadati un elemento alla volta
            def process_metadata_async():
                global DOMANDE
                global metadata_processing_status
                
                try:
                    # Aggiorniamo lo stato di elaborazione
                    metadata_processing_status['in_progress'] = True
                    metadata_processing_status['total_questions'] = len(valid_items)
                    metadata_processing_status['processed_questions'] = 0
                    metadata_processing_status['start_time'] = datetime.now()
                    metadata_processing_status['error'] = None
                    
                    logger.info("Avvio generazione metadati per le domande - un elemento alla volta...")
                    
                    # Estrai i testi delle domande
                    question_texts = []
                    for q in valid_items:
                        if "Domanda" in q:
                            question_texts.append(q["Domanda"])
                        elif "text" in q:
                            question_texts.append(q["text"])
                    
                    if not question_texts:
                        logger.error("Nessun testo di domanda disponibile per la generazione dei metadati")
                        return
                    
                    # Creiamo un file temporaneo JSON con le domande
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False, encoding='utf-8') as tmp:
                        json.dump(question_texts, tmp)
                        temp_file_path = tmp.name
                        logger.info(f"File temporaneo creato: {temp_file_path}")
                    
                    try:
                        # Chiamiamo direttamente la funzione di generazione metadati di QuestionImporter
                        logger.info(f"Chiamata a QuestionImporter.generate_metadata per {len(question_texts)} domande")
                        question_metas = QuestionImporter.generate_metadata(temp_file_path)
                        logger.info(f"Generati metadati per {len(question_metas)} domande")
                        
                        # Inizializza la struttura per il salvataggio dei metadati
                        metadata_dict = {
                            'timestamp': datetime.now().isoformat(),
                            'total_questions': len(question_metas),
                            'questions': []
                        }
                        
                        # Aggiorniamo le domande con i metadati mantenendo l'ordine originale - UN ELEMENTO ALLA VOLTA
                        for i, meta in enumerate(question_metas):
                            if i < len(valid_items) and i < len(DOMANDE):
                                # Metadati correnti
                                primary_topic = meta.primary_topic
                                subtopics = meta.subtopics
                                keywords = meta.keywords
                                
                                # Aggiorna il dizionario DOMANDE con i metadati disponibili
                                DOMANDE[i]['topic'] = primary_topic
                                DOMANDE[i]['subtopics'] = subtopics
                                DOMANDE[i]['keywords'] = keywords
                                DOMANDE[i]['lemma_sets'] = meta.lemma_sets
                                DOMANDE[i]['fuzzy_norms'] = meta.fuzzy_norms
                                DOMANDE[i]['vectors'] = meta.vectors
                                
                                # Aggiorna anche gli item originali per retrocompatibilità
                                valid_items[i]["topics"] = [primary_topic] + subtopics
                                valid_items[i]["keywords"] = keywords
                                
                                # Log dei risultati
                                question_id = DOMANDE[i].get("id", f"q{i}")
                                logger.info(f"Metadati generati per domanda {i+1}: {DOMANDE[i]['testo'][:50]}...")
                                logger.info(f"  Topic: {primary_topic}, Subtopics: {subtopics}")
                                
                                # Prepara i dati per il file JSON
                                question_data = {
                                    'id': question_id,
                                    'domanda': DOMANDE[i]['testo'][:100] + ('...' if len(DOMANDE[i]['testo']) > 100 else ''),
                                    'primary_topic': primary_topic,
                                    'subtopics': subtopics,
                                    'keywords': keywords,
                                    'lemma_sets': meta.lemma_sets,
                                    'fuzzy_norms': meta.fuzzy_norms,
                                    'vectors': meta.vectors
                                }
                                metadata_dict['questions'].append(question_data)
                                
                                # Aggiornamento contatore
                                metadata_processing_status['processed_questions'] += 1
                                
                                # Breve pausa per non sovraccaricare il sistema
                                time.sleep(0.01)
                        
                        # Aggiornamento stato finale
                        metadata_processing_status['in_progress'] = False
                        metadata_processing_status['end_time'] = datetime.now()
                        logger.info(f"Completata generazione metadati per {len(question_metas)} domande")
                        
                        # Salva i metadati in un file JSON per visualizzazione
                        try:
                            metadata_file_path = os.path.join(os.path.dirname(temp_file_path), 'metadati_generati.json')
                            
                            # Scrivi il file JSON con indentazione per leggibilità
                            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                                json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
                                
                            logger.info(f"Metadati salvati in file JSON: {metadata_file_path}")
                        except Exception as e:
                            logger.error(f"Errore nel salvataggio dei metadati in file JSON: {e}")
                    except Exception as e:
                        metadata_processing_status['error'] = str(e)
                        logger.error(f"Errore nella generazione dei metadati con QuestionImporter: {e}")
                    finally:
                        # Pulizia del file temporaneo
                        try:
                            os.unlink(temp_file_path)
                            logger.debug(f"File temporaneo rimosso: {temp_file_path}")
                        except Exception as e:
                            logger.warning(f"Impossibile rimuovere il file temporaneo: {e}")
                except Exception as e:
                    metadata_processing_status['error'] = str(e)
                    logger.error(f"Errore generale nella generazione dei metadati: {e}")
            
            # Avvia l'elaborazione metadati in un thread separato per non bloccare
            thread = threading.Thread(target=process_metadata_async)
            thread.daemon = True  # Il thread muore quando il programma principale termina
            thread.start()
            logger.info("Thread di elaborazione metadati avviato con successo")
        else:
            logger.warning("QuestionImporter non disponibile, metadati non verranno generati")
        
        return True
    except Exception as e:
        logger.error(f"Errore durante il caricamento dello script: {e}", exc_info=True)
        return False

def get_session_info() -> Dict[str, Any]:
    """Restituisce informazioni sulle sessioni attive"""
    return {
        "active_sessions": len(SESSIONS),
        "session_ids": list(SESSIONS.keys()),
        "script_size": len(SCRIPT)
    }

def get_question_metadata_status(question_id: str) -> Dict[str, Any]:
    """
    Verifica lo stato dei metadati di una domanda specifica nella struttura DOMANDE
    
    Args:
        question_id: ID della domanda da verificare
        
    Returns:
        Dizionario con lo stato dei metadati e, se disponibili, i metadati stessi
    """
    # Cerca la domanda nella nuova struttura DOMANDE per indice
    found_index = -1
    for i, q in enumerate(DOMANDE):
        if q.get('id') == question_id:
            found_index = i
            break
    
    if found_index >= 0:
        # La domanda esiste nella struttura DOMANDE
        domanda = DOMANDE[found_index]
        
        # Verifica se i metadati sono stati generati
        if domanda.get('topic') and domanda.get('subtopics'):
            # I metadati sono stati generati
            return {
                'status': 'completed',
                'metadata': {
                    'primary_topic': domanda.get('topic', ''),
                    'subtopics': domanda.get('subtopics', []),
                    'keywords': domanda.get('keywords', []),
                    'lemma_sets': domanda.get('lemma_sets', []),
                    'fuzzy_norms': domanda.get('fuzzy_norms', []),
                    'vectors': domanda.get('vectors', [])
                },
                'message': f"Metadati per domanda {question_id} completati"
            }
        else:
            # La domanda esiste ma i metadati non sono ancora pronti
            return {
                'status': 'pending',
                'metadata': None,
                'message': 'Metadati in elaborazione'
            }
    else:
        # Se non trovata in DOMANDE, cerca in SCRIPT per retrocompatibilità
        found = False
        for q in SCRIPT:
            if q.get('id') == question_id:
                found = True
                break
        
        if found:
            # La domanda esiste in SCRIPT ma non in DOMANDE
            logger.warning(f"Domanda {question_id} trovata in SCRIPT ma non in DOMANDE")
            return {
                'status': 'pending',
                'metadata': None,
                'message': 'Metadati non ancora generati - usa SCRIPT per retrocompatibilità'
            }
        else:
            # La domanda non esiste in nessuna struttura
            logger.warning(f"Domanda {question_id} non trovata in DOMANDE né in SCRIPT")
            return {
                'status': 'not_found',
                'metadata': None,
                'message': 'Domanda non trovata'
            }
        
def wait_for_question_ready(question_id: str, 
                          max_attempts: int = 2, 
                          initial_delay: float = 1.0,
                          backoff_factor: float = 1.5) -> Optional[Dict[str, Any]]:
    """
    Attende che i metadati di una domanda siano pronti con backoff esponenziale.
    Limitato a un massimo di 2 tentativi come richiesto.
    Se i metadati non sono pronti dopo i tentativi, si restituisce comunque lo stato
    per permettere all'applicazione di procedere con la domanda successiva.
    
    Args:
        question_id: ID della domanda da controllare
        max_attempts: Numero massimo di tentativi (default: 2)
        initial_delay: Ritardo iniziale in secondi (default: 1.0)
        backoff_factor: Fattore di moltiplicazione per il backoff (default: 1.5)
        
    Returns:
        Lo stato della domanda, anche se i metadati non sono pronti
    """
    delay = initial_delay
    for attempt in range(max_attempts):
        status = get_question_metadata_status(question_id)
        
        if status['status'] == 'completed':
            logger.info(f"Metadati per domanda {question_id} pronti al tentativo {attempt + 1}")
            return status
        elif status['status'] == 'not_found':
            logger.error(f"Domanda {question_id} non trovata nel sistema")
            return None
            
        # Aspetta con backoff esponenziale
        logger.info(f"Attesa metadati per domanda {question_id} (tentativo {attempt + 1}/{max_attempts})...")
        time.sleep(delay)
        delay *= backoff_factor  # Aumenta il ritardo per il prossimo tentativo
    
    logger.warning(f"Timeout attesa metadati per domanda {question_id} dopo {max_attempts} tentativi. Procedo con la domanda successiva.")
    return status  # Restituisce comunque lo stato anche se non è 'completed'

# Funzione per ottenere la prossima domanda disponibile
def get_next_available_question(current_question_id: str, max_attempts: int = 2) -> Optional[str]:
    """
    Trova la prossima domanda disponibile nella struttura DOMANDE.
    Utilizza un counter e un ciclo come richiesto per navigare tra le domande.
    Se i metadati non sono pronti entro il timeout, procede comunque alla domanda successiva.
    
    Args:
        current_question_id: ID della domanda corrente
        max_attempts: Numero massimo di tentativi per domanda
        
    Returns:
        ID della prossima domanda disponibile, o None se non ce ne sono altre
    """
    # Verifichiamo se ci sono domande nella struttura DOMANDE
    if not DOMANDE:
        logger.warning("Struttura DOMANDE vuota, nessuna domanda disponibile")
        return None
    
    # Trova l'indice della domanda corrente in DOMANDE
    current_index = -1
    for i, q in enumerate(DOMANDE):
        if q.get("id") == current_question_id:
            current_index = i
            break
    
    # Se la domanda corrente non è stata trovata o è l'ultima
    if current_index == -1 or current_index >= len(DOMANDE) - 1:
        # Se la domanda non è stata trovata ma ci sono domande, restituisci la prima
        if current_index == -1 and len(DOMANDE) > 0:
            logger.warning(f"Domanda corrente {current_question_id} non trovata in DOMANDE, restituisco la prima domanda")
            return DOMANDE[0].get("id")
        else:
            logger.info("Siamo all'ultima domanda, nessuna domanda successiva disponibile")
            return None
    
    # Ottieni il numero totale di domande
    numero_domande = len(DOMANDE)
    
    # Inizializza un contatore per navigare attraverso le domande
    i = current_index + 1  # Inizia dalla domanda successiva a quella corrente
    
    # Utilizza un ciclo per trovare la prossima domanda disponibile
    # Se stiamo cercando la domanda i-esima (i > 0), otteniamo DOMANDE[i]
    if i < numero_domande:
        next_question = DOMANDE[i]
        next_id = next_question.get("id")
        
        if next_id:
            logger.info(f"Prossima domanda: {next_id} (indice {i})")
            
            # Verifica se i metadati sono pronti con timeout
            counter = 0
            while counter < max_attempts:
                # Controlla se i metadati sono pronti
                if next_question.get("topic") and next_question.get("subtopics"):
                    logger.info(f"Metadati per la domanda {next_id} sono pronti")
                    break
                
                # Altrimenti aspetta un po' e riprova
                logger.info(f"Attesa metadati per domanda {next_id} (tentativo {counter + 1}/{max_attempts})")
                time.sleep(1.0)  # Attendi 1 secondo tra i tentativi
                counter += 1
            
            # Anche se i metadati non sono pronti dopo i tentativi, procedi comunque
            if counter >= max_attempts:
                logger.warning(f"Timeout attesa metadati per domanda {next_id}. Procedo comunque.")
            
            return next_id
    
    return None

def get_metadata_processing_status() -> Dict[str, Any]:
    """
    Restituisce lo stato attuale dell'elaborazione dei metadati.
    Utilizza la nuova struttura DOMANDE per calcolare lo stato attuale.
    
    Returns:
        Dizionario con informazioni sullo stato dell'elaborazione
    """
    global metadata_processing_status
    global DOMANDE
    
    # Base result dal dizionario di stato
    result = metadata_processing_status.copy()
    
    # Se DOMANDE contiene elementi, ricalcoliamo lo stato effettivo di elaborazione
    if DOMANDE:
        # Conta quante domande hanno effettivamente metadati completi
        processed_count = 0
        for domanda in DOMANDE:
            if domanda.get('topic') and domanda.get('subtopics'):
                processed_count += 1
        
        # Aggiorna il contatore in base ai dati effettivi
        total_questions = len(DOMANDE)
        result['total_questions'] = total_questions
        result['processed_questions'] = processed_count
        
        # Aggiorna flag in_progress
        result['in_progress'] = processed_count < total_questions
        
        # Se tutti i metadati sono stati elaborati ma non era stato segnalato
        if processed_count == total_questions and result['in_progress']:
            result['in_progress'] = False
            result['end_time'] = datetime.now()
    
    # Calcola la percentuale di completamento
    if result['total_questions'] > 0:
        result['completion_percentage'] = (result['processed_questions'] / result['total_questions']) * 100
    else:
        result['completion_percentage'] = 0
    
    # Calcola il tempo trascorso se l'elaborazione è in corso
    if result['in_progress'] and result['start_time']:
        elapsed = datetime.now() - result['start_time']
        result['elapsed_seconds'] = elapsed.total_seconds()
    elif result['end_time'] and result['start_time']:
        elapsed = result['end_time'] - result['start_time']
        result['elapsed_seconds'] = elapsed.total_seconds()
    else:
        result['elapsed_seconds'] = 0
    
    # Aggiungi informazioni sulla struttura DOMANDE
    result['domande_structure'] = {
        'count': len(DOMANDE),
        'status': 'active' if DOMANDE else 'empty'
    }
    
    # Converti i datetime in stringhe per la serializzazione JSON
    if result['start_time']:
        result['start_time'] = result['start_time'].isoformat()
    if result['end_time']:
        result['end_time'] = result['end_time'].isoformat()
    
    return result
