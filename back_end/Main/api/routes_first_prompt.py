from fastapi import APIRouter, HTTPException, Query, Depends, Cookie, Request
from typing import Dict, Any, Optional
import logging
import base64
from jose import JWTError, jwt

# Import dai moduli interni
from Main.core import config
from Main.core.config import DEVELOPMENT_MODE
from Main.models import TTSResponse, ErrorResponse
from Main.application.user_session_service import SCRIPT, get_state
from .auth import get_current_user

# Configurazione logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API router
router = APIRouter(tags=["FirstPrompt"])

@router.get("/first_prompt", response_model=Dict[str, Any], responses={
    401: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def get_first_prompt(
    request: Request,
    user_id: str = Query(..., description="ID dell'utente"),
    token_query: Optional[str] = Query(None, description="Token JWT in query parameter"),
    token: Optional[str] = Cookie(None)  # Usiamo il cookie opzionale invece di Depends
):
    """Restituisce il primo prompt audio per iniziare l'intervista."""
    try:
        logger.info(f"Richiesta primo prompt per utente: {user_id}")
        
        # Verifica e decodifica il token, sia da cookie che da query parameter
        try:
            current_user = None
            
            # Forza modalità di sviluppo per questa sessione (test e debug)
            force_dev_mode = True  # <-- Importante: usato solo per debugging
            
            # Prendo il token da query parameter o da cookie
            actual_token = token_query if token_query else token
            logger.info(f"Token: {'Presente in query' if token_query else 'Presente in cookie' if token else 'Non presente'}")
            
            # Controlla sia DEVELOPMENT_MODE che il flag force_dev_mode
            is_dev_mode = config.DEVELOPMENT_MODE or force_dev_mode
            
            # In modalità sviluppo, permetti l'accesso senza token o con token fittizio
            if is_dev_mode:
                logger.info("Modalità sviluppo o forzata: permesso accesso senza autenticazione completa")
                
                # Se c'è un token fittizio che inizia con 'dev_token_', NON tentare di decodificarlo come JWT
                if actual_token and actual_token.startswith('dev_token_'):
                    logger.info(f"Dev token rilevato: {actual_token[:20]}... - Accesso sviluppo consentito")
                    current_user = user_id
                else:
                    logger.info(f"Accesso sviluppo: usando user_id={user_id} direttamente")
                    current_user = user_id  # Usa l'ID utente fornito direttamente
                
                # In modalità sviluppo, saltiamo completamente la logica JWT
                logger.info(f"Dev mode: autenticazione bypass per user_id={user_id}")
            elif not actual_token:
                # Solo in produzione richiediamo obbligatoriamente un token
                logger.error("Nessun token fornito (né in cookie né in query parameter)")
                raise HTTPException(status_code=401, detail="Token non fornito")
            else:
                # Modalità produzione: verifica token reale JWT
                logger.info(f"Modalità produzione: decodifica token JWT: {actual_token[:20]}...")
                try:
                    payload = jwt.decode(actual_token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
                    current_user = payload.get("sub")
                    
                    if not current_user:
                        logger.error(f"Token decodificato, ma 'sub' mancante: {payload}")
                        raise HTTPException(status_code=401, detail="Token senza identificatore utente")
                        
                    logger.info(f"Auth OK: user={current_user}, richiesto per session={user_id}")
                except Exception as jwt_err:
                    logger.error(f"Errore durante la decodifica del token: {jwt_err}")
                    raise HTTPException(status_code=401, detail=f"Token non valido: {jwt_err}")
        except JWTError as e:
            # Questo catch viene eseguito solo se si tenta di decodificare un token JWT non valido
            is_dev_mode = config.DEVELOPMENT_MODE or force_dev_mode
            if is_dev_mode:
                logger.warning(f"JWTError in modalità sviluppo: {e}, continuiamo con user_id={user_id}")
                current_user = user_id
            else:
                logger.error(f"JWTError: {e}")
                raise HTTPException(status_code=401, detail=f"Token non valido: {e}")
        except Exception as general_e:
            # Errore generale di autenticazione
            is_dev_mode = config.DEVELOPMENT_MODE or force_dev_mode
            logger.error(f"Errore generale durante l'autenticazione: {general_e}")
            if is_dev_mode:
                logger.warning(f"Continuiamo in modalità sviluppo con user_id={user_id}")
                current_user = user_id
            else:
                raise HTTPException(status_code=500, detail=f"Errore di autenticazione: {str(general_e)}")
        
        # Ottieni la sessione dell'utente o creane una nuova
        logger.info(f"Recupero sessione per user_id: {current_user}")
        session = get_state(current_user)
        
        # Controlla se le domande sono state caricate
        if not session.questions or len(session.questions) == 0:
            # Se non ci sono domande nella sessione, verifica se ci sono domande globali in SCRIPT
            if SCRIPT and len(SCRIPT) > 0:
                logger.info(f"Nessuna domanda nella sessione, ma SCRIPT globale ha {len(SCRIPT)} domande. Le uso per questa sessione.")
                session.questions = SCRIPT  # Copia le domande globali nella sessione
            else:
                logger.info(f"Nessuna domanda caricata per user: {current_user}, restituisco messaggio informativo")
                return {
                    "message": "Nessuna domanda caricata. Caricare le domande prima di iniziare il colloquio.",
                    "audio_url": None,
                    "question_text": "Nessuna domanda disponibile",
                    "question_type": "info",
                    "question_index": 0,
                    "questions_loaded": False
                }
        
        # Preparare il testo introduttivo per l'intervista
        intro_text = "Benvenuto all'intervista X. Mettiti comodo e rilassati.Sono qui per farti alcune domande. Non è un esame, quindi non ci sono domande giuste o sbagliate. Detto questo, iniziamo con la prima domanda: "
        
        # Utilizza la nuova struttura DOMANDE per ottenere la prima domanda
        from Main.application.user_session_service import DOMANDE
        
        # Verifichiamo se ci sono domande nella struttura DOMANDE
        if DOMANDE and len(DOMANDE) > 0:
            # Prendiamo la prima domanda dalla struttura DOMANDE
            first_question_data = DOMANDE[0]
            question_id = first_question_data.get("id", "q1")
            
            # Accediamo direttamente al testo della domanda dalla struttura DOMANDE
            question_text = first_question_data.get("testo", "Chi sei e quali sono le tue competenze principali?")
            logger.info(f"Prima domanda ottenuta da DOMANDE: {question_text[:50]}...")
        else:
            # Fallback alla vecchia struttura se DOMANDE non contiene elementi
            logger.warning("Struttura DOMANDE vuota, uso il vecchio metodo con SCRIPT")
            first_question = session.questions[0]
            question_id = first_question.get("id", "q1")
            question_text = first_question.get("Domanda", first_question.get("text", "Chi sei e quali sono le tue competenze principali?"))
            logger.info(f"Prima domanda ottenuta da SCRIPT: {question_text[:50]}...")

        
        # Componi il testo completo da convertire in audio
        full_text = f"{intro_text} {question_text}"
        
        # Non generiamo più l'audio direttamente qui, utilizziamo solo l'endpoint TTS interno
        voice = "Bianca"  # Usa la voce italiana predefinita per il TTS interno
        logger.info(f"Utilizzo TTS interno per il testo: '{full_text[:50]}...' con voce {voice}")
        
        # Non generiamo né codifichiamo l'audio, sarà gestito dall'endpoint TTS
        audio_content = ""  # Nessun audio embedded
        logger.info("Audio sarà generato tramite endpoint TTS separato")
            
        # Prepara URL per il TTS con i parametri necessari (voce e testo)
        import urllib.parse
        encoded_text = urllib.parse.quote(full_text)
        tts_url = f"{config.API_BASE_URL}/api/tts/speak?voice_id={voice}&text={encoded_text}"
        
        logger.info(f"Primo prompt pronto per l'utente {current_user}: {full_text[:50]}...")
        
        return {
            "status": "success",
            "message": "Primo prompt generato con successo",
            "text": full_text,
            "audio_content": audio_content,  # Audio già generato in base64
            "audio_url": tts_url,  # URL del servizio TTS (backup)
            "question_id": question_id
        }
        
    except Exception as e:
        logger.error(f"Errore nella generazione del primo prompt: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Errore nella generazione del primo prompt: {str(e)}"
        )
        
@router.get("/load_questions_status")
async def load_questions_status():
    """Controlla se ci sono domande caricate nel sistema."""
    script_size = len(SCRIPT)
    first_question = None
    questions_loaded = script_size > 0
    
    if questions_loaded and SCRIPT[0]:
        first_question = SCRIPT[0].get("Domanda", "<Domanda non trovata>")[:50] + "..."
    
    # Controlla anche le sessioni attive
    from Main.application.user_session_service import get_session_info
    session_info = get_session_info()
    
    return {
        "status": "success" if questions_loaded else "warning",
        "questions_loaded": questions_loaded,
        "question_count": script_size,
        "first_question_preview": first_question,
        "active_sessions": session_info["active_sessions"],
        "dev_mode": config.DEVELOPMENT_MODE
    }
