from fastapi import APIRouter, Depends, HTTPException, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordRequestForm
from typing import Optional
from pydantic import BaseModel
from jose import jwt, JWTError
from datetime import datetime, timedelta
import logging

from Main.core import config

# Configurazione logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelli Pydantic per le risposte
class Token(BaseModel):
    access_token: str
    token_type: str
    
class SessionStatus(BaseModel):
    valid: bool
    message: Optional[str] = None
    questions_loaded: Optional[bool] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
# Variabili di configurazione dal modulo centralizzato
JWT_SECRET = config.JWT_SECRET
JWT_ALGORITHM = config.JWT_ALGORITHM
TOKEN_EXPIRATION_DAYS = config.JWT_EXPIRATION_DAYS

# Funzione per recuperare il token dalla richiesta
def get_token_from_request(request) -> Optional[str]:
    """Recupera il token JWT dalla richiesta, sia dai cookie che dagli header."""
    try:
        # Prova a prendere il token dall'header Authorization
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split("Bearer ")[1].strip()
            logger.info(f"Token trovato nell'header Authorization: {token[:20]}...")
            return token
        
        # Prova a prendere il token dal cookie
        token = request.cookies.get("token")
        if token:
            logger.info(f"Token trovato nel cookie: {token[:20]}...")
            return token
            
        return None
    except Exception as e:
        logger.error(f"Errore durante il recupero del token: {e}")
        return None

# Security bearer setup
security = HTTPBearer()

# API router
router = APIRouter(tags=["Authentication"])

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Crea un token JWT con i dati specificati e durata opzionale."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get the current user ID from the token. Ensures 'sub' exists.
    Supporta anche token fittizi per lo sviluppo (dev_token_)."""
    try:
        token = credentials.credentials
        
        # Supporto per token di sviluppo (compatibile con interview.js e home.js)
        if token.startswith('dev_token_'):
            logger.info(f"Riconosciuto token di sviluppo: {token[:20]}...")
            # Estrai l'ID utente dal token di sviluppo (formato: dev_token_BASE64(userId)_timestamp)
            parts = token.split('_')
            if len(parts) >= 3:
                try:
                    # Estrai l'ID utente in base64
                    import base64
                    user_id = base64.b64decode(parts[2]).decode('utf-8')
                    logger.info(f"ID utente estratto dal token di sviluppo: {user_id}")
                    return user_id
                except Exception as ex:
                    # Se c'è un errore nell'estrazione, usa il valore predefinito
                    logger.warning(f"Errore nell'estrazione dell'ID dal token di sviluppo: {ex}")
                    return config.AUTH_DEV_USERNAME
            # Se il formato non è corretto, usa l'utente di sviluppo predefinito
            return config.AUTH_DEV_USERNAME
        
        # Gestione JWT standard
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            logger.warning(f"Invalid token: 'sub' claim missing in payload: {payload}")
            raise HTTPException(status_code=401, detail="Invalid token: User identifier missing")
        return user_id
    except JWTError as e:
        logger.warning(f"JWTError during token validation: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
        
async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[str]:
    """Versione opzionale di get_current_user che non genera errori se il token manca.
    Utile per endpoint che supportano sia utenti autenticati che non."""
    if not credentials:
        logger.info("Nessun token fornito (opzionale)")
        return None
    
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            logger.warning(f"Token senza 'sub' claim")
            return None
        return user_id
    except Exception as e:
        logger.warning(f"Errore nell'autenticazione opzionale: {e}")
        return None

@router.post("/token", response_model=None, responses={200: {"model": Token}, 401: {"model": None}})
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # DEVELOPMENT MODE: Accetta solo le credenziali admin/admin
    # In un'applicazione reale, dovresti validare le credenziali contro un database
    
    # Credenziali fisse per lo sviluppo - manteniamo le preferenze dell'utente (admin/admin)
    DEV_USERNAME = config.AUTH_DEV_USERNAME
    DEV_PASSWORD = config.AUTH_DEV_PASSWORD
    
    user_id = form_data.username
    password = form_data.password
    
    logger.info(f"Token requested for user: {user_id}")
    
    # Verifica le credenziali fisse
    if user_id != DEV_USERNAME or password != DEV_PASSWORD:
        logger.warning(f"Login failed: Invalid credentials for user '{user_id}'")
        raise HTTPException(
            status_code=401,
            detail="Credenziali non valide",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info(f"Login successful for development user: {user_id}")
    
    # Genera il token con un tempo di scadenza più lungo per lo sviluppo
    access_token_expires = timedelta(days=TOKEN_EXPIRATION_DAYS) # Usa i giorni invece dei minuti per sviluppo
    access_token = create_access_token(
        data={"sub": user_id, "questions_loaded": False}, # Inizialmente nessuna domanda caricata
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# Funzione di utilitu00e0 per aggiornare lo stato di caricamento delle domande nel token
def mark_questions_loaded(token, loaded=True):
    """Aggiorna il token JWT per indicare che le domande sono state caricate"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        # Aggiorna il flag questions_loaded mantenendo gli altri dati
        payload["questions_loaded"] = loaded
        # Ricrea il token con i dati aggiornati
        updated_token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        logger.info(f"Token aggiornato per user {payload.get('sub')}: questions_loaded={loaded}")
        return updated_token
    except Exception as e:
        logger.error(f"Errore nell'aggiornamento del token: {e}")
        return token  # Ritorna il token originale in caso di errore

@router.get("/check_session", response_model=None, responses={200: {"model": SessionStatus}})
async def check_session(token: Optional[str] = Cookie(None), token_query: Optional[str] = None):
    """Controlla se una sessione ha domande valide caricate.
    Accetta il token sia dal cookie che dal parametro di query 'token'."""
    active_token = token_query if token_query else token
    if not active_token:
        logger.info("check_session: No active token found.")
        return SessionStatus(valid=False, message="No session token provided")
    try:
        payload = jwt.decode(active_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            logger.warning(f"check_session: Token valid but 'sub' (user_id) missing. Payload: {payload}")
            return SessionStatus(valid=False, message="Invalid session: User identifier missing")
        questions_loaded = payload.get("questions_loaded", False)
        logger.info(f"check_session for user_id '{user_id}': Token valid, questions_loaded={questions_loaded}")
        # Verifichiamo se le domande sono veramente caricate nel sistema
        from Main.application.user_session_service import SCRIPT
        questions_really_loaded = len(SCRIPT) > 0
        
        # Se abbiamo questions_loaded=True nel token MA lo SCRIPT è vuoto,
        # c'è un problema di sincronizzazione
        if questions_loaded and not questions_really_loaded:
            logger.error(f"ERRORE: Token indica questions_loaded=True ma SCRIPT è vuoto! Correggo...")
            questions_loaded = False  # Correggo l'incongruenza
        
        # Aggiungiamo debug dettagliato
        logger.info(f"[DEBUG] check_session - token: {questions_loaded}, SCRIPT: {len(SCRIPT)} domande")
        
        return SessionStatus(
            valid=True,
            questions_loaded=questions_loaded if questions_really_loaded else False,
            session_id=active_token,
            user_id=user_id
        )
    except JWTError as e:
        logger.warning(f"check_session: JWTError - {e}. Token: {active_token[:20]}...")
        return SessionStatus(valid=False, message="Invalid or expired session")
