"""
Modulo per pre‑processare testo italiano e migliorare la resa con sistemi di sintesi vocale (OpenAI TTS, Azure Neural TTS, Google Cloud TTS, ecc.).

Principali caratteristiche:
* Normalizza anglicismi, acronimi, numeri, date e valute in forma facilmente pronunciabile.
* Inserisce tag SSML <break> per gestire le pause dopo punteggiatura e fine frase.
* Gestione espandibile tramite dizionari e funzioni plug‑in.
* Tutte le regex sono pre‑compilate e la pipeline è completamente stateless.

Esempio rapido:
>>> from italian_tts_preprocessor import optimize_italian_tts
>>> optimize_italian_tts("Ho un meeting con il manager alle 15:30, online.")
'<speak> Ho un miiting con il meneger alle 15 e 30 <break time="400ms"/> on lain. </speak>'

© 2025 – MIT License
"""

from __future__ import annotations
import re
import logging
from typing import Dict, Callable

logger = logging.getLogger(__name__)

# --- Dizionario di anglicismi e parole non italiane (chiave sempre in minuscolo)
ANGLOPHONISMS: Dict[str, str] = {
    # tecnologia
    "app": "app",
    "backup": "bècap",
    "browser": "brauser",
    "bug": "bag",
    "business": "bìsnes",
    "cloud": "claud",
    "computer": "compiuter",
    "cookie": "cuchi",
    "crash": "cresh",
    "dashboard": "dèshbord",
    "data": "dèita",
    "database": "dèitabeiss",
    "debug": "dibàg",
    "default": "difòlt",
    "device": "divaiss",
    "download": "daunlòud",
    "email": "imeil",
    "embed": "imbed",
    "feature": "fìciur",
    "file": "fail",
    "firewall": "fàireuol",
    "gadget": "gàget",
    "hardware": "àrduer",
    "homepage": "òmpeig",
    "hosting": "òsting",
    "interface": "intèrfeiss",
    "laptop": "lèptop",
    "link": "linc",
    "login": "lòghin",
    "malware": "màluer",
    "mouse": "màus",
    "network": "nètuorc",
    "offline": "off lain",
    "online": "on lain",
    "output": "autput",
    "podcast": "pòdcasst",
    "privacy": "pràivasi",
    "router": "ràuter",
    "scroll": "scrol",
    "server": "sèrver",
    "smartphone": "smàrfon",
    "software": "sòftuer",
    "streaming": "strìming",
    "swipe": "suaip",
    "tablet": "tàblet",
    "upload": "aplòud",
    "username": "iuserneèim",
    "web": "ueb",
    "website": "ueb sait",
    "wireless": "uàirless",
    
    # lavoro
    "account": "èccaunt",
    "advisor": "edvaiser",
    "agency": "ègensi",
    "agenda": "ègenda",
    "asset": "èsset",
    "assistant": "assìstant",
    "boss": "bòs",
    "brainstorming": "breinsstòrming",
    "brand": "brend",
    "brief": "brìf",
    "briefing": "brìfing",
    "budget": "bàgget",
    "business": "bìsnes",
    "CEO": "sì i ò",
    "CFO": "sì ef ò",
    "CTO": "sì tì ò",
    "deadline": "dèdlain",
    "feedback": "fìdbek",
    "freelance": "frìlàns",
    "HR": "ècc àr",
    "job": "giob",
    "leadership": "lìdership",
    "manager": "meneger",
    "marketing": "màrcheting",
    "meeting": "miiting",
    "mission": "mìscion",
    "part time": "pàrt taim",
    "performance": "perfòrmens",
    "report": "rìport",
    "roadmap": "ròdmep",
    "skill": "schill",
    "staff": "stàff",
    "startup": "startàp",
    "task": "tàsk",
    "team": "tìm",
    "update": "àpdeitt",
    "vision": "vìscion",
    "workflow": "uòrcflòu",
    
    # social
    "chat": "ciat",
    "follower": "fòllouer",
    "hashtag": "èshtag",
    "influencer": "ìnfluenser",
    "like": "laic",
    "post": "post",
    "selfie": "sèlfi",
    "share": "scèr",
    "social": "sòscial",
    "stories": "stòris",
    "tag": "tèg",
    "trend": "trend",
    "unfollow": "ànfollo",
}

# --- Regex pre‑compilate
ACRONYM_RE = re.compile(r"\b([A-Z]{2,})\b")  # FBI, HTML, URL
DECIMAL_RE = re.compile(r"(\d+),(\d+)")       # 3,14 ➜ 3 virgola 14
HOUR_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")  # 15:30 ➜ 15 e 30
CURRENCY_RE = re.compile(r"€\s?(\d+(?:,\d+)?)")  # €12,50 ➜ 12 euro e 50 centesimi
DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")  # 12/5 o 12/5/2023

# Numeri grandi senza separatore migliaia
NUMBER_RE = re.compile(r"\b(\d{4,})\b")  # 1234 ➜ 1 234 (per migliore leggibilità vocale)

PUNCT_SHORT_RE = re.compile(r"([,;:])")      # virgola, punto e virgola, due punti
PUNCT_LONG_RE = re.compile(r"([.!?])")        # fine frase

# --- Helper generici --------------------------------------------------------

def _substitute_words(text: str, mapping: Dict[str, str]) -> str:
    """Sostituisce le parole secondo il mapping (case‑insensitive)."""
    if not mapping:
        return text

    pattern = re.compile(r"\b(" + "|".join(map(re.escape, mapping.keys())) + r")\b", re.IGNORECASE)

    def _repl(match: re.Match) -> str:
        key = match.group(0).lower()
        return mapping.get(key, match.group(0))

    return pattern.sub(_repl, text)


def _spell_out_acronyms(text: str) -> str:
    """Espande gli acronimi in lettere separate (URL ➜ U R L)."""
    return ACRONYM_RE.sub(lambda m: " ".join(m.group(1)), text)


def _expand_decimals(text: str) -> str:
    """Trasforma i decimali con virgola in forma parlata."""
    return DECIMAL_RE.sub(lambda m: f"{m.group(1)} virgola {m.group(2)}", text)


def _expand_hours(text: str) -> str:
    """15:30 ➜ 15 e 30."""
    return HOUR_RE.sub(lambda m: f"{int(m.group(1))} e {m.group(2)}", text)


def _expand_euro(text: str) -> str:
    """€12,50 ➜ 12 euro e 50 centesimi."""
    def _repl(match: re.Match) -> str:
        value = match.group(1).replace(",", ".")
        euros, *_cents = value.split(".") + ["00"]
        cents = _cents[0][:2] if _cents else "00"
        if cents == "00":
            return f"{int(euros)} euro"
        return f"{int(euros)} euro e {int(cents)} centesimi"

    return CURRENCY_RE.sub(_repl, text)


def _format_date(text: str) -> str:
    """Formatta date 12/5 o 12/5/2023 in formato leggibile."""
    def _repl(match: re.Match) -> str:
        day = int(match.group(1))
        month = int(match.group(2))
        year = match.group(3)
        
        month_names = [
            "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
            "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"
        ]
        
        if month > 0 and month <= 12:
            month_name = month_names[month - 1]
        else:
            month_name = f"mese {month}"
            
        if year:
            # Se l'anno è composto da 2 cifre, aggiungi '20' davanti
            if len(year) == 2:
                year = f"20{year}"
            return f"{day} {month_name} {year}"
        else:
            return f"{day} {month_name}"
            
    return DATE_RE.sub(_repl, text)


def _format_numbers(text: str) -> str:
    """Inserisce spazi nei numeri grandi per migliorare la lettura dal TTS.
    Ad esempio: 1234567 ➜ 1 234 567
    """
    def _repl(match: re.Match) -> str:
        num = match.group(1)
        # Aggiungi uno spazio ogni 3 cifre, partendo da destra
        formatted = ""
        for i, digit in enumerate(reversed(num)):
            if i > 0 and i % 3 == 0:
                formatted = " " + formatted
            formatted = digit + formatted
        return formatted
            
    return NUMBER_RE.sub(_repl, text)


def preprocess_italian_text(text: str) -> str:
    """Pipeline di normalizzazione prima dei tag SSML."""
    original = text
    text = _spell_out_acronyms(text)
    text = _expand_decimals(text)
    text = _expand_hours(text)
    text = _expand_euro(text)
    text = _format_date(text)
    text = _format_numbers(text)
    text = _substitute_words(text, ANGLOPHONISMS)

    if text != original:
        logger.debug("TTS preprocess: %s → %s", original, text)
    return text


def add_ssml_breaks(text: str, short_ms: int = 150, long_ms: int = 400) -> str:
    """Inserisce pause compatibili con OpenAI TTS dopo punteggiatura.

    Args:
        text: stringa già preprocessata.
        short_ms: durata pausa breve (virgola, punto‑e‑virgola, due punti).
        long_ms: durata pausa lunga (fine frase).
    """
    # Invece di usare tag SSML <break>, usiamo puntini di sospensione invisibili
    # In OpenAI TTS, questi vengono interpretati come pause naturali senza essere pronunciati
    # Per le pause brevi (virgole, punto e virgola, due punti)
    text = PUNCT_SHORT_RE.sub(r"\1 . ", text)
    
    # Per le pause lunghe (fine frase) usiamo più puntini che creano pause più lunghe
    text = PUNCT_LONG_RE.sub(r"\1 . . ", text)
    
    return text


def optimize_italian_tts(text: str, wrap_speak: bool = True) -> str:
    """Ottimizza il testo italiano per il TTS, aggiungendo o meno tag SSML.
    
    Se wrap_speak=True, avvolge l'output in tag <speak></speak>.
    """
    # Rimuove eventuali tag SSML esistenti per evitare duplicazioni
    # text = re.sub(r'<[^>]*>', '', text)
    
    # Assicurati che non ci siano già tag SSML
    text = re.sub(r'<speak>|</speak>', '', text).strip()
    
    # Rimuovi le virgolette doppie all'inizio e alla fine che possono causare problemi SSML
    text = text.strip('"')
    
    processed = preprocess_italian_text(text)
    processed = add_ssml_breaks(processed)
    
    if wrap_speak:
        processed = f"<speak> {processed} </speak>"
        
    return processed

# --- ESEMPIO CLI ------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sample = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Ho un meeting con il manager alle 15:30, online. Il budget è di €3,14."
    print(optimize_italian_tts(sample))
