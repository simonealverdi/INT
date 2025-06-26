from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import Dict, List, Any, Optional
import logging
import tempfile
import os
from pydantic import BaseModel

# Configurazione logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelli di dati per API
class QuestionItem(BaseModel):
    id: str
    text: str
    category: Optional[str] = None
    difficulty: Optional[str] = None

class QuestionResponse(BaseModel):
    status: str
    message: str
    count: int

class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str

# Storage temporaneo per le domande (in produzione useremmo un database)
questions_db = []

# Router per le domande
router = APIRouter(tags=["Questions"])

@router.post("/load", response_model=None, responses={
    200: {"model": QuestionResponse},
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def load_questions(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint per caricare domande da un file CSV (semplificato per sviluppo)"""
    try:
        # Simula il caricamento di alcune domande esempio
        # In una versione reale, qui leggeremmo il contenuto del file
        global questions_db
        
        # Log della richiesta
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        logger.info(f"Ricevuto file '{file.filename}' di {file_size} bytes")
        
        # Salva il file UploadFile in un file temporaneo
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            from Importazioni import QuestionImporter
            questions_text = QuestionImporter.import_questions(tmp_path)
            questions_db.clear()
            
            # Creiamo una lista di domande nel formato corretto per lo SCRIPT
            from Main.application.user_session_service import SCRIPT, load_script
            formatted_questions = []
            
            for idx, qtext in enumerate(questions_text, 1):
                # Aggiungi alla lista locale per l'API
                questions_db.append(QuestionItem(id=f"q{idx}", text=qtext))
                
                # Aggiungi allo SCRIPT globale nel formato corretto
                formatted_questions.append({
                    "id": f"q{idx}",
                    "Domanda": qtext,
                    "Tipologia": "personalizzata"
                })
            
            # Carica le domande formattate nello SCRIPT globale
            if formatted_questions:
                # Aggiungere subito le domande allo SCRIPT globale (senza attendere l'elaborazione dei metadati)
                from Main.application.user_session_service import get_metadata_processing_status
                success = load_script(formatted_questions)
                
                # Ottenere lo stato dell'elaborazione dei metadati
                metadata_status = get_metadata_processing_status()
                processing_info = {
                    'metadata_processing': True,
                    'total_questions': metadata_status.get('total_questions', 0),
                    'processed_questions': metadata_status.get('processed_questions', 0)
                }
                
                if success:
                    logger.info(f"SCRIPT globale aggiornato con {len(formatted_questions)} domande - Metadati in elaborazione")
                else:
                    logger.warning("Impossibile aggiornare lo SCRIPT globale")
            
            logger.info(f"Caricate {len(questions_db)} domande da file {file.filename}")
            # Rimuovi il file temporaneo
            os.remove(tmp_path)
            return {
                "status": "success",
                "message": f"Caricate {len(questions_db)} domande dal file. Puoi accedere a interview.html immediatamente.",
                "count": len(questions_db),
                "metadata_processing": {
                    "in_progress": processing_info.get('metadata_processing', False),
                    "total_questions": processing_info.get('total_questions', 0),
                    "processed_questions": processing_info.get('processed_questions', 0),
                    "note": "I metadati verranno elaborati in background. Puoi controllare lo stato tramite /questions/metadata-status"
                }
            }
        except Exception as e:
            logger.error(f"Errore durante l'importazione delle domande: {e}")
            from fastapi.responses import JSONResponse
            # Rimuovi il file temporaneo anche in caso di errore
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return JSONResponse({"status": "error", "message": f"Errore durante l'importazione: {str(e)}"}, status_code=400)
        
    except HTTPException as he:
        logger.error(f"Errore HTTP {he.status_code} durante il caricamento domande: {he.detail}")
        from fastapi.responses import JSONResponse
        return JSONResponse({"status": "error", "message": str(he.detail)}, status_code=he.status_code)
    except Exception as e:
        logger.error(f"Errore durante il caricamento delle domande: {e}")
        from fastapi.responses import JSONResponse
        return JSONResponse({"status": "error", "message": f"Errore durante il caricamento: {str(e)}"}, status_code=500)

@router.get("/list", response_model=None, responses={
    200: {"model": QuestionResponse}
})
async def list_questions() -> Dict[str, Any]:
    """Ottieni l'elenco di tutte le domande disponibili"""
    return {
        "status": "success",
        "count": len(questions_db),
        "questions": questions_db
    }

@router.get("/random", response_model=None, responses={
    200: {"model": QuestionResponse},
    404: {"model": ErrorResponse}
})
async def get_random_question() -> Dict[str, Any]:
    """Ottieni una domanda casuale dal database"""
    import random
    
    if not questions_db:
        raise HTTPException(status_code=404, detail="Nessuna domanda disponibile. Carica prima le domande.")
    
    # Seleziona una domanda casuale
    question = random.choice(questions_db)
    
    return {
        "status": "success",
        "question": question
    }

@router.get("/count", response_model=None)
async def get_question_count() -> Dict[str, int]:
    """Ottieni il numero di domande disponibili"""
    return {"count": len(questions_db)}

@router.get("/metadata-status", response_model=None)
async def get_metadata_status() -> Dict[str, Any]:
    """Ottieni lo stato dell'elaborazione dei metadati per le domande caricate"""
    try:
        # Importa la funzione di monitoraggio dei metadati
        from Main.application.user_session_service import get_metadata_processing_status
        
        # Ottieni lo stato corrente
        status = get_metadata_processing_status()
        
        # Costruisci la risposta
        return {
            "status": "success",
            "message": "Stato di elaborazione metadati ottenuto con successo",
            "metadata_processing": {
                "in_progress": status.get('in_progress', False),
                "total_questions": status.get('total_questions', 0),
                "processed_questions": status.get('processed_questions', 0),
                "completion_percentage": status.get('completion_percentage', 0),
                "elapsed_seconds": status.get('elapsed_seconds', 0),
                "domande_structure": status.get('domande_structure', {})
            }
        }
    except Exception as e:
        logger.error(f"Errore nel recupero dello stato dei metadati: {e}")
        return {
            "status": "error",
            "message": f"Errore nel recupero dello stato dei metadati: {str(e)}",
            "metadata_processing": {
                "in_progress": False,
                "error": str(e)
            }
        }
