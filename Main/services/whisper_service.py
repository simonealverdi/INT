from typing import BinaryIO
from Main.core.logger import logger
from Main.core import config
import tempfile, os, time, asyncio
from openai import AsyncOpenAI

# Client OpenAI singleton
async_client = AsyncOpenAI()

async def speech_to_text_openai(audio_bytes: bytes) -> str:
    """Trascrive audio in testo usando l'API OpenAI Whisper.
    Accetta bytes audio e restituisce il testo trascritto.
    """
    t0 = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_audio_file:
            tmp_audio_file.write(audio_bytes)
            tmp_audio_path = tmp_audio_file.name
        
        try:
            with open(tmp_audio_path, 'rb') as audio_file_opened:
                transcript = await async_client.audio.transcriptions.create(
                    file=audio_file_opened,
                    model="whisper-1",
                    language="it",
                    response_format="text"
                )
            text = str(transcript).strip()
            logger.info(f"OpenAI Whisper API took {time.perf_counter() - t0:.2f}s for transcription")
            return text
        finally:
            os.unlink(tmp_audio_path) # Assicurati di rimuovere il file temporaneo
    except Exception as e:
        logger.error(f"OpenAI Whisper API error durante la trascrizione: {e}")
        raise

# -----------------------------------------------------------------------------
# Utilità di trascrizione secondarie
# -----------------------------------------------------------------------------


async def _process_audio_with_openai_whisper(audio_file: BinaryIO) -> str:
    """Usa OpenAI Whisper API (large-v3 via model whisper-1)."""
    t0 = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_audio_file:
            tmp_audio_file.write(audio_file.read())
            tmp_audio_path = tmp_audio_file.name
        
        try:
            with open(tmp_audio_path, 'rb') as audio_file:
                transcript = await async_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",  # Modello Whisper di OpenAI
                    language="it",  # Italiano
                    response_format="text"
                )
            return str(transcript)
        finally:
            os.unlink(tmp_audio_path)
    except Exception as e:
        logger.error(f"OpenAI Whisper API error durante la trascrizione: {e}")
        raise


