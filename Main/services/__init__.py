# Export dei servizi dell'applicazione

# Import dei servizi principali
try:
    from Main.services.tts_service import text_to_speech, stream_tts
    from Main.services.whisper_service import speech_to_text_openai
    from Main.services.llm_service import _call_gpt, async_call_gpt, generate_llm_clarification_request
except ImportError as e:
    import sys
    print(f"Errore di importazione nei servizi: {e}", file=sys.stderr)
