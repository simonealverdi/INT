import os
import uuid
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Optional

# Caricamento del file .env (necessario solo se non caricato in precedenza)
load_dotenv()

# -----------------------------------------------------------------------------
# Modalità di sviluppo e configurazioni generali
# -----------------------------------------------------------------------------

# Modalità di sviluppo: se True, vengono usati dati fittizi e skip di alcune API
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "true").lower() in ("true", "1", "yes")

# Impostazioni di logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "app.log")

# Crea la directory dei log se non esiste
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# URL base dell'API per le chiamate interne ed esterne
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# -----------------------------------------------------------------------------
# Percorsi e struttura file
# -----------------------------------------------------------------------------

# Percorsi principali
BACK_END_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Cartelle di cache
TTS_CACHE_DIR = os.path.join(BACK_END_ROOT, "tts_cache")
AUDIO_CACHE_DIR = os.path.join(BACK_END_ROOT, "audio")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Credenziali e configurazioni API
# -----------------------------------------------------------------------------

# OpenAI - Modelli e parametri
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_REFLECT_MODEL = os.getenv("OPENAI_REFLECT_MODEL", "gpt-4o")
OPENAI_TEMPERATURE_REFLECTION = float(os.getenv("TEMP_REFLECTION", "0.3"))
OPENAI_TEMPERATURE_EXPERT = float(os.getenv("TEMP_EXPERT", "0.4"))
OPENAI_TEMPERATURE_FOLLOWUP = float(os.getenv("TEMP_FOLLOWUP", "0.6"))

# OpenAI TTS
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1-hd")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "nova")  # alloy, echo, fable, onyx, nova, shimmer
OPENAI_TTS_FORMAT = os.getenv("OPENAI_TTS_FORMAT", "mp3")
OPENAI_TTS_SPEED = float(os.getenv("OPENAI_TTS_SPEED", "1.0"))

# AWS Polly
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_POLLY_VOICE_ID = os.getenv("AWS_POLLY_VOICE_ID", "Bianca")
AWS_POLLY_ENGINE = os.getenv("AWS_POLLY_ENGINE", "neural")
AWS_POLLY_FORMAT = os.getenv("AWS_POLLY_FORMAT", "mp3")

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_ENABLED = bool(MONGODB_URI) and not DEVELOPMENT_MODE

# -----------------------------------------------------------------------------
# Autenticazione e sicurezza
# -----------------------------------------------------------------------------

# JWT
JWT_SECRET = os.getenv("JWT_SECRET", str(uuid.uuid4()))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
JWT_EXPIRATION_DAYS = int(os.getenv("TOKEN_EXPIRATION_DAYS", "7"))

# Credenziali di sviluppo (fisse per facilità in fase di sviluppo)
AUTH_DEV_USERNAME = "admin"  # Manteniamo 'admin' come preferito dall'utente
AUTH_DEV_PASSWORD = "admin"  # Manteniamo 'admin' come preferito dall'utente

# -----------------------------------------------------------------------------
# Moduli esterni
# -----------------------------------------------------------------------------

# Aggiungi il percorso BACK_END_ROOT al sys.path per importazioni esterne
if BACK_END_ROOT not in sys.path:
    sys.path.append(BACK_END_ROOT)

# -----------------------------------------------------------------------------
# Helper generici
# -----------------------------------------------------------------------------

def get_env(key: str, default: Any = None, cast: Any = str) -> Any:
    """
    Recupera una variabile d'ambiente con conversione di tipo.
    
    Args:
        key: Nome della variabile d'ambiente
        default: Valore di default
        cast: Funzione di conversione (str, int, float, bool)
        
    Returns:
        Valore convertito o default in caso di errore
    """
    val = os.getenv(key, default)
    try:
        return cast(val)
    except Exception:
        return default

def ensure_dir_exists(path: str) -> None:
    """
    Assicura che una directory esista, creandola se necessario.
    """
    os.makedirs(path, exist_ok=True)
