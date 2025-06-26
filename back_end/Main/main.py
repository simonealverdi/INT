
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
import uvicorn
import atexit

# Configurazione centralizzata
from Main.core import config
from Main.services.persistence_service import dump_dev_storage_to_file

# Configurazione di logging centralizzata
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Imposta variabili di ambiente e path
os.environ['PYTHONPATH'] = os.path.join(os.getcwd())

# Salvataggio dei dati di sviluppo alla chiusura dell'applicazione
if config.DEVELOPMENT_MODE:
    logger.info("Modalità sviluppo attiva - i dati saranno salvati alla chiusura")
    atexit.register(dump_dev_storage_to_file)

# Importa i router dalla struttura a livelli
try:
    # Importa i router principali
    from Main.api import auth_router, tts_router, questions_router, first_prompt_router, transcribe_router
    
    # Importa interview_router dal nuovo modulo di compatibilità
    from Main.api.interview_router import router as interview_router
except ImportError as e:
    import sys
    print(f"Errore di importazione nei router API: {e}", file=sys.stderr)
    # Crea stub per evitare errori di avvio
    from fastapi import APIRouter
    auth_router = APIRouter()
    tts_router = APIRouter()
    questions_router = APIRouter()
    interview_router = APIRouter()
    first_prompt_router = APIRouter()
    transcribe_router = APIRouter()

# Inizializzazione dell'app FastAPI
app = FastAPI(
    title="AI Interview API",
    description="API per l'intervista condotta con intelligenza artificiale",
    version="1.0.0"
)

# Configurazione CORS - permissiva in modalità sviluppo, più restrittiva altrimenti
origins = ["*"] if config.DEVELOPMENT_MODE else [
    "http://localhost:3000",    # React app principale
    "http://localhost:8080",
    "http://localhost:5173",    # Eventuale alternativa
    "https://aiint.app"         # Dominio di produzione
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

# Log delle impostazioni principali
logger.info(f"Modalità sviluppo: {config.DEVELOPMENT_MODE}")
logger.info(f"MongoDB abilitato: {config.MONGODB_ENABLED}")
logger.info(f"CORS origins: {origins}")

# Registra i router
app.include_router(auth_router, prefix="/api", tags=["auth"])
app.include_router(tts_router, prefix="/api/tts", tags=["tts"])
app.include_router(questions_router, prefix="/api/questions", tags=["questions"])
app.include_router(interview_router, prefix="/api/interview", tags=["interview"])
app.include_router(first_prompt_router, prefix="/api", tags=["first_prompt"])
app.include_router(transcribe_router, prefix="/api", tags=["transcribe"])

# Endpoint informativi
@app.get("/")
async def root():
    return {
        "name": "AI Interview API",
        "version": "1.0.0",
        "status": "online - in fase di sviluppo",
        "endpoints": [
            "/api/token - Autenticazione (username: admin, password: admin)",
            "/api/check-token - Verifica validità token",
            "/api/tts/speak - Sintesi vocale (versione semplificata)",
            "/api/tts/status - Stato del servizio TTS",
            "/api/questions/load - Carica domande (mock)",
            "/api/questions/list - Elenco domande disponibili",
            "/api/questions/random - Ottieni domanda casuale",
            "/api/questions/count - Numero di domande disponibili",
            "/api/interview/start - Avvia una nuova intervista",
            "/api/interview/next-question/{interview_id} - Ottieni prossima domanda",
            "/api/interview/submit-answer/{interview_id}/{question_id} - Invia risposta",
            "/api/interview/status/{interview_id} - Stato intervista",
            "/api/interview/end/{interview_id} - Termina intervista"
        ],
        "nota": "Implementazione incrementale - funzionalità in espansione"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ---------------------------------------------------------
# Funzione di avvio di sviluppo
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        # Assicurati che tutte le cartelle necessarie esistano
        config.ensure_dir_exists(config.TTS_CACHE_DIR)
        config.ensure_dir_exists(config.AUDIO_CACHE_DIR)
        
        # Crea directory per i dump di sviluppo
        if config.DEVELOPMENT_MODE:
            dump_dir = os.path.join(config.BACK_END_ROOT, "data_dumps")
            config.ensure_dir_exists(dump_dir)
        
        # Log dei dettagli dell'applicazione
        logger.info("=== DETTAGLI DELL'APPLICAZIONE ===")
        logger.info(f"Router registrati: auth, tts, questions, interview")
        logger.info(f"OPENAI_MODEL: {config.OPENAI_MODEL}")
        logger.info(f"OPENAI API Key configurata: {bool(config.OPENAI_API_KEY)}")
        logger.info(f"TTS model: {config.OPENAI_TTS_MODEL}, voice: {config.OPENAI_TTS_VOICE}")
        logger.info(f"Directory di backend: {config.BACK_END_ROOT}")
        
        # Avvia server con reload in modalità sviluppo
        logger.info("Avvio dell'applicazione...")
        uvicorn.run(
            "Main.main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=config.DEVELOPMENT_MODE,  # Ricarica automatica solo in sviluppo
            log_level="debug" if config.DEVELOPMENT_MODE else "info"
        )
    except Exception as e:
        logger.exception(f"Errore durante l'avvio dell'applicazione: {e}")
        # Mostra dettagli completi dello stack trace
        import traceback
        logger.error(traceback.format_exc())
