from typing import Dict, Optional, Any, List, Union, Callable, Tuple
import asyncio
import json
import logging
import time

from Main.core import config
from openai import OpenAI, AsyncOpenAI

# Configurazione logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utilizziamo la configurazione centralizzata

# Client OpenAI singleton inizializzato solo quando necessario
def get_openai_client():
    """Ottiene un'istanza di client OpenAI con configurazione centralizzata."""
    if config.DEVELOPMENT_MODE and not config.OPENAI_API_KEY:
        # In modalità sviluppo senza chiave API, ritorna un client che fallirà gentilmente
        logger.warning("Modalità sviluppo: nessuna chiave API OpenAI configurata")
        return OpenAI(api_key="sk-dummy-key-for-development")
    
    try:
        # Utilizza la chiave API dalla configurazione centralizzata
        return OpenAI(api_key=config.OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"Non è stato possibile creare un client OpenAI: {e}")
        return None
        
def get_async_openai_client():
    """Ottiene un'istanza di client asincrono OpenAI con configurazione centralizzata."""
    if config.DEVELOPMENT_MODE and not config.OPENAI_API_KEY:
        # In modalità sviluppo senza chiave API, ritorna un client che fallirà gentilmente
        logger.warning("Modalità sviluppo: nessuna chiave API OpenAI configurata")
        return AsyncOpenAI(api_key="sk-dummy-key-for-development")
    
    try:
        # Utilizza la chiave API dalla configurazione centralizzata
        return AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"Non è stato possibile creare un client asincrono OpenAI: {e}")
        return None

def _call_gpt(messages: list[dict], *, temperature: float = None, max_tokens: int = 150, model: str = None) -> str:
    """Wrapper sincrono che restituisce direttamente il content strip.
    In modalità sviluppo, può restituire risposte simulate senza chiamare API."""
    t0 = time.perf_counter()
    
    # Usa i parametri dalla configurazione centralizzata
    model = model or config.OPENAI_MODEL
    temperature = temperature if temperature is not None else config.OPENAI_TEMPERATURE_EXPERT
    
    # Ottieni client OpenAI
    client = get_openai_client()
    
    # Modalità sviluppo: se non c'è client o siamo in modalità sviluppo senza API key
    if client is None or (config.DEVELOPMENT_MODE and not config.OPENAI_API_KEY):
        logger.info(f"Utilizzo modalità sviluppo per la chiamata LLM: model={model}, temp={temperature}")
        
        # Simula risposte in base al tipo di messaggio e ai contenuti
        first_user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
        system_message = next((m["content"] for m in messages if m["role"] == "system"), "")
        
        # Risposte simulate più contestuali in base al tipo di richiesta
        if "follow" in first_user_message.lower() or "follow-up" in first_user_message.lower() or "approfondire" in first_user_message.lower():
            missing_topics = ["ruoli precedenti", "competenze tecniche", "obiettivi di carriera"]
            return f"Potresti parlarmi più in dettaglio di: {', '.join(missing_topics[:2])}?".strip()
            
        elif "clarif" in first_user_message.lower() or "chiar" in first_user_message.lower():
            return "Non ho compreso completamente la tua risposta. Potresti spiegare meglio cosa intendi?".strip()
            
        elif "analis" in first_user_message.lower() or "proced" in first_user_message.lower():
            return "Basandomi sulla tua risposta, potremmo approfondire la tua esperienza con [tecnologia rilevante] e come l'hai applicata in progetti precedenti.".strip()
            
        else:
            return "Questa è una risposta simulata generata in modalità sviluppo senza utilizzo di API esterne.".strip()
    
    # Modalitu00e0 normale: usa l'API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = response.choices[0].message.content.strip()
        logger.info(f"GPT call to {model} took {time.perf_counter() - t0:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Error in GPT call: {e}")
        # Restituisci una risposta di fallback invece di lanciare eccezione
        return "Non u00e8 stato possibile generare una risposta. Riprova piu00f9 tardi."



async def async_call_gpt(messages: list[dict], *, temperature: float = None, max_tokens: int = 150, model: str = None) -> str:
    """Wrapper asincrono che restituisce direttamente il content strip.
    In modalità sviluppo, può restituire risposte simulate senza chiamare API."""
    t0 = time.perf_counter()
    
    # Usa i parametri dalla configurazione centralizzata
    model = model or config.OPENAI_MODEL
    temperature = temperature if temperature is not None else config.OPENAI_TEMPERATURE_EXPERT
    
    # Ottieni client OpenAI asincrono
    async_client = get_async_openai_client()
    
    # Modalità sviluppo: se non c'è client o siamo in modalità sviluppo senza API key
    if async_client is None or (config.DEVELOPMENT_MODE and not config.OPENAI_API_KEY):
        logger.info(f"Utilizzo modalità sviluppo per chiamata LLM asincrona: model={model}, temp={temperature}")
        
        # Simula risposte in base al tipo di messaggio e ai contenuti
        first_user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
        system_message = next((m["content"] for m in messages if m["role"] == "system"), "")
        
        # Risposte simulate più contestuali in base al tipo di richiesta
        if "follow" in first_user_message.lower() or "follow-up" in first_user_message.lower() or "approfondire" in first_user_message.lower():
            missing_topics = ["esperienze professionali", "progetti completati", "competenze tecniche"]
            return f"Mi piacerebbe saperne di più su {', '.join(missing_topics[:2])}. Potresti approfondire questi aspetti?"
            
        elif "clarif" in first_user_message.lower() or "chiar" in first_user_message.lower():
            return "Scusa, non sono sicuro di aver capito completamente la tua risposta. Potresti spiegare meglio il tuo punto di vista?"
            
        elif "analis" in first_user_message.lower() or "valut" in first_user_message.lower():
            return "Basandomi sulla tua risposta, noto che hai menzionato alcuni punti interessanti. Potresti approfondire come questi aspetti si collegano alla posizione per cui ti stai candidando?"
            
        else:
            return f"[MODALITÀ SVILUPPO] Risposta simulata per: {first_user_message[:50]}..."
            
        
    # Modalità normale: usa l'API
    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = response.choices[0].message.content.strip()
        logger.info(f"GPT call to {model} took {time.perf_counter() - t0:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Error in GPT async call: {e}")
        # Restituisci una risposta di fallback invece di lanciare eccezione
        return "Non è stato possibile generare una risposta. Riprova più tardi."



def _is_valid_followup(q: str) -> bool:
    """Ritorna True se la domanda rispetta criteri minimi di qualità."""
    return bool(q and q.endswith("?") and 5 <= len(q) <= 120)



async def _follow_up_async(
    question: str, 
    transcript: str, 
    reflection_notes: str, 
    missing_subtopics: List[str]  # NUOVO PARAMETRO
 ) -> str:
    """Genera UNA domanda di follow-up naturale partendo dalla domanda corrente.

    Il modello riceve:
        • Domanda principale (assistant)
        • Trascrizione risposta utente (user)
        • Eventuali note/reflection (assistant)
        • Lista dei subtopic mancanti (per potenziale uso nel prompt, non usato direttamente ora)

    Viene poi istruito (system) a produrre soltanto la domanda.
    Viene applicata una validazione leggera; se fallisce si effettua 1 retry.
    """
    few_shot_examples = [
        {
            "role": "system",
            "content": (
                "Esempio 1 — DOMANDA: Qual è la tua giornata tipo?\n"
                "RISPOSTA: Di solito mi sveglio alle 7, porto i bambini a scuola e poi lavoro in ufficio.\n"
                "FOLLOW-UP: Qual è il momento più impegnativo della tua giornata?"
            ),
        },
        {
            "role": "system",
            "content": (
                "Esempio 2 — DOMANDA: Che sport pratichi?\n"
                "RISPOSTA: Mi piace andare a correre due volte a settimana.\n"
                "FOLLOW-UP: Cosa ti motiva a mantenere questa routine?"
            ),
        },
    ]

    # Prepara la stringa dei subtopic mancanti per il prompt
    subtopics_mancanti_str = ", ".join(missing_subtopics) if missing_subtopics else "nessun aspetto specifico"
    
    # Selezione del subtopic prioritario per il follow-up
    subtopic_to_target = missing_subtopics[0] if missing_subtopics else "aspetto non approfondito"

    # Prompt di sistema che include i subtopic mancanti
    system_prompt_content = (
        f"Sei un intervistatore HR italiano, cortese e curioso.\n"
        f"L'utente ha appena risposto alla domanda principale. Dalla sua risposta, "
        f"sembra che i seguenti argomenti/aspetti chiave non siano stati toccati o approfonditi a sufficienza: "
        f"{subtopics_mancanti_str}.\n\n"
        f"Formula UNA sola domanda di follow-up, massimo 25 parole, in italiano colloquiale.\n"
        f"La domanda deve:\n"
        f"1. Terminare con '?'\n"
        f"2. Essere naturale e collegata alla precedente risposta dell'utente.\n"
        f"3. Invitare l'utente ad approfondire specificamente sul subtopic '{subtopic_to_target}', collegandoti se possibile alla risposta precedente.\n"
        f"Evita di chiedere genericamente \"puoi dirmi di più?\". Sii più specifico, riferendoti a uno dei temi non trattati."
    )

    # Costruisci i messaggi base dinamicamente per includere il system_prompt_content aggiornato
    # e rimuovere il vecchio messaggio di sistema per il follow-up.
    dynamic_base_messages = [
        {"role": "system", "content": "Sei un intervistatore HR italiano, cortese e curioso."},
        *few_shot_examples,
        {"role": "assistant", "content": question},  # La domanda principale
        {"role": "user", "content": transcript},     # La risposta dell'utente
    ]
    if reflection_notes:
        dynamic_base_messages.append({"role": "assistant", "content": f"NOTE: {reflection_notes}"})
    
    dynamic_base_messages.append({"role": "system", "content": system_prompt_content})

    # Tentiamo fino a 2 volte (1 retry) se la validazione fallisce
    for attempt in range(2):
        try:
            candidate = await async_call_gpt(
                dynamic_base_messages, # Usa i messaggi costruiti dinamicamente
                temperature=config.OPENAI_TEMPERATURE_FOLLOWUP,
                max_tokens=60, # Max tokens per la domanda di follow-up
            )
            if _is_valid_followup(candidate):
                return candidate
            # altrimenti aggiungiamo feedback e riproviamo
            # Aggiungi il feedback ai messaggi dinamici
            dynamic_base_messages.append({
                "role": "system",
                "content": "La precedente risposta non rispettava i requisiti. Riprova generando UNA domanda breve che termini con '?'",
            })
        except Exception as e:
            logger.error(f"Error generating follow-up (attempt {attempt+1}): {e}")

    # Fallback
    return "Potresti raccontarmi qualcosa di più a riguardo?"



async def generate_llm_clarification_request(question_text: str, llm_client=None) -> str:
    """Genera un messaggio di chiarimento quando la trascrizione di una risposta fallisce.
    Versione semplificata per sviluppo che non richiede chiamate API esterne."""
    logger.info(f"Generazione messaggio di chiarimento per domanda: '{question_text[:50]}...'")
    
    # Se non abbiamo il testo della domanda, usiamo un riferimento generico
    if not question_text:
        question_text = "la tua risposta precedente"
    
    # Versione di sviluppo: messaggio preformattato invece di chiamare l'API
    # Rotazione di diversi messaggi per simulare variabilità
    import random
    clarification_templates = [
        "Non sono riuscito a capire bene la tua risposta riguardo a '{question}'. Potresti ripeterla?",
        "Mi dispiace, la registrazione non era chiara. Potresti ridire la tua risposta su '{question}'?",
        "Potresti ripetere la tua risposta su '{question}', per favore? Non l'ho afferrata.",
        "Scusa, potresti dirlo un'altra volta? Non ho colto la tua risposta su '{question}'.",
        "Non ho sentito bene. Puoi ripetere la tua risposta alla domanda su '{question}'?"
    ]
    
    # Seleziona un template casuale e inserisci la domanda
    template = random.choice(clarification_templates)
    clarification_message = template.format(question=question_text)
    
    logger.info(f"Messaggio di chiarimento generato: {clarification_message}")
    return clarification_message


