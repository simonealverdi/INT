from fastapi import APIRouter, Depends, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import tempfile, shutil, os, json
from typing import List, Dict, Any, Optional
from datetime import timedelta

# Import dai moduli interni
from Main.core.logger import logger
from Main.core import config
from Main.api.auth import create_access_token, get_current_user
from Main.application.user_session_service import load_script, SESSIONS
from Main.api.models import QuestionResponse, ErrorResponse

# Import di helper esterni
import sys
sys.path.append(config.BACK_END_ROOT)
from Importazioni import QuestionImporter

# API router
router = APIRouter(tags=["Questions"])

@router.post("/load_questions", response_model=None, responses={200: {"model": QuestionResponse}, 400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def load_questions(file: UploadFile = File(...), background_tasks: Optional[BackgroundTasks] = None, current_user: str = Depends(get_current_user)):
    """Carica domande da un file (supporta .docx, .csv, .xls, .json)
    e popola SCRIPT con metadati, inclusi domande e topic."""
    try:
        logger.info(f"User '{current_user}' is attempting to load questions from file: {file.filename}, type: {file.content_type}")

        # Salva il file caricato temporaneamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        logger.info(f"File temporaneo salvato in: {tmp_path}")

        # Carica le domande e genera i metadati usando QuestionImporter (supporta .docx, .csv, .xls, .json)
        try:
            question_metadata_list = QuestionImporter.generate_metadata(tmp_path)
            logger.info(f"Domande importate con successo: {len(question_metadata_list)} trovate.")
            if question_metadata_list:
                logger.info(f"Esempio prima domanda: {question_metadata_list[0].prompt[:50]}...")
        except Exception as e:
            logger.error(f"Errore durante l'importazione delle domande da '{tmp_path}': {e}", exc_info=True)
            # Rimuovi il file temporaneo in caso di errore di importazione, se esiste ancora
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as remove_err:
                    logger.error(f'Failed to remove temp file {tmp_path} after import error: {remove_err}')
            raise ValueError(f"Errore durante l'importazione delle domande: {e}")

        # Trasforma i metadati nel formato SCRIPT atteso da InterviewState
        processed_script = []
        for meta_item in question_metadata_list:
            topics_for_interview = meta_item.subtopics if meta_item.subtopics else ([meta_item.primary_topic] if meta_item.primary_topic else [])
            new_question = {"Domanda": meta_item.prompt, "Tipologia": "main", "Subtopics": topics_for_interview}
            processed_script.append(new_question)

        # Carica lo script nel servizio - aggiungiamo debug dettagliato
        success = load_script(processed_script)
        logger.info(f"Script processato con successo, {len(processed_script)} domande convertite. Load_script restituisce: {success}")
        
        # DEBUG: verifichiamo lo stato globale dopo il caricamento
        from Main.application.user_session_service import SCRIPT
        logger.info(f"[DEBUG] Stato globale SCRIPT dopo caricamento: contiene {len(SCRIPT)} domande")
        if len(SCRIPT) > 0:
            logger.info(f"[DEBUG] Prima domanda nello SCRIPT: {SCRIPT[0].get('Domanda', '')[:50]}...")
        else:
            logger.error("[DEBUG] ERRORE CRITICO: SCRIPT è vuoto dopo caricamento!")
        
        # Rimuovi il file temporaneo in background o direttamente
        if background_tasks:
            background_tasks.add_task(os.remove, tmp_path)
        else:
            try:
                os.remove(tmp_path) # Fallback se non ci sono background tasks
            except Exception as e:
                logger.warning(f'Failed to remove temp file {tmp_path} immediately: {e}')

        # Creiamo semplicemente un nuovo token con questions_loaded=True
        # Questo risolve il problema della sessione invalida
        from Main.api.auth import create_access_token
        session_token_expires = timedelta(days=config.TOKEN_EXPIRATION_DAYS)
        updated_token = create_access_token(
            data={"sub": current_user, "questions_loaded": True}, 
            expires_delta=session_token_expires
        )
        logger.info(f"Creato nuovo token con questions_loaded=True: {updated_token[:20]}...")

        response_data = {
            "status": "success",
            "count": len(processed_script),
            "first_question": processed_script[0]["Domanda"] if processed_script else None,
            "session_token": updated_token
        }
        
        # DEBUG: stampiamo informazioni dettagliate
        logger.info(f"[DEBUG] Creando risposta con token: {updated_token[:20]}...")
        
        response = JSONResponse(response_data)

        # Imposta il cookie con il token di sessione - Aggiungiamo opzioni per assicurare che il cookie venga salvato
        # e sia accessibile sia da frontend che backend
        cookie_max_age = 60 * 60 * 24 * 7  # 7 giorni in secondi
        response.set_cookie(
            key="token",
            value=updated_token,
            httponly=False,  # False durante sviluppo per consentire accesso JavaScript
            max_age=cookie_max_age,
            secure=False,  # False per sviluppo (http://localhost)
            samesite="none" if config.DEVELOPMENT_MODE else "lax"  # 'none' durante sviluppo per cross-origin
        )
        
        # DEBUG: Verifichiamo che il cookie sia stato impostato
        logger.info(f"[DEBUG] Cookie 'token' impostato con valore (troncato): {updated_token[:20]}..., max_age={cookie_max_age}")
        logger.info(f"Domande caricate con successo per user '{current_user}'. Session token impostato.")
        return response
    except ValueError as ve:
        logger.error(f"Errore di validazione o importazione durante il caricamento delle domande per user '{current_user}': {ve}", exc_info=True)
        return JSONResponse({"status": "error", "message": str(ve)}, status_code=400)
    except Exception as e:
        logger.error(f"Errore generico durante il caricamento delle domande per user '{current_user}': {e}", exc_info=True)
        # Assicurati che tmp_path sia definito per il cleanup, anche se potrebbe non esserlo in caso di errori molto precoci
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as remove_err:
                logger.error(f'Failed to remove temp file {tmp_path} during generic error handling: {remove_err}')
        return JSONResponse({"status": "error", "message": f"Errore interno del server: {e}"}, status_code=500)
