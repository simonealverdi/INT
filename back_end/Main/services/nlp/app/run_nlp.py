from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caricamento del modello spaCy
try:
    logger.info("Caricamento del modello spaCy it_core_news_sm...")
    nlp = spacy.load("it_core_news_sm")
    logger.info("Modello spaCy caricato con successo")
except Exception as e:
    logger.error(f"Errore nel caricamento del modello spaCy: {e}")
    raise

# Caricamento del modello SBERT
try:
    logger.info("Caricamento del modello SBERT...")
    # Utilizziamo un modello più recente e sicuramente disponibile
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Modello SBERT caricato con successo")
except Exception as e:
    logger.error(f"Errore nel caricamento del modello SBERT: {e}")
    sbert = None
    # Fallback: utilizza embedding medi
    logger.warning("Fallback a embedding medi semplici - performance ridotte")

app = FastAPI(title="NLP Service", description="Servizio NLP per l'analisi del testo italiano")

class Text(BaseModel):
    text: str

class CosineSimilarityRequest(BaseModel):
    vector1: list[float]
    vector2: list[float]

class NormalizeVectorRequest(BaseModel):
    vector: list[float]

@app.get("/")
async def root():
    return {"status": "online", "service": "NLP Service", "endpoints": ["/parse", "/cosine_similarity", "/normalize_vector"]}

@app.post("/parse")
async def parse_text(data: Text):
    try:
        logger.info(f"Elaborazione testo: {data.text[:50]}...")
        doc = nlp(data.text)
        
        # Estrazione dei token con lemmi
        tokens = [{"text": t.text, "lemma": t.lemma_} for t in doc]
        
        # Estrazione delle entità
        entities = [(e.text, e.label_) for e in doc.ents]
        
        # Estrazione del vettore del documento
        vector = doc.vector.tolist()
        
        # Se SBERT è disponibile, usa quello per un vettore migliore
        if sbert:
            vector = sbert.encode(data.text, normalize_embeddings=True).tolist()
        
        logger.info(f"Elaborazione completata: {len(tokens)} token, {len(entities)} entità")
        return {
            "tokens": tokens,
            "entities": entities,
            "vector": vector
        }
    except Exception as e:
        logger.error(f"Errore nell'elaborazione del testo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cosine_similarity")
async def cosine_similarity(data: CosineSimilarityRequest):
    try:
        vec1 = np.array(data.vector1)
        vec2 = np.array(data.vector2)
        similarity = np.dot(vec1, vec2)
        return {"similarity": float(similarity)}
    except Exception as e:
        logger.error(f"Errore nel calcolo della similarità coseno: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/normalize_vector")
async def normalize_vector(data: NormalizeVectorRequest):
    try:
        vec = np.array(data.vector)
        norm = np.linalg.norm(vec)
        if norm > 0:
            normalized = vec / norm
        else:
            normalized = vec
        return {"normalized": normalized.tolist()}
    except Exception as e:
        logger.error(f"Errore nella normalizzazione del vettore: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Avvio del servizio NLP sulla porta 9000...")
    uvicorn.run(app, host="0.0.0.0", port=9000)