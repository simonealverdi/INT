# Export dei router API

# Importazione dei router - percorsi relativi per evitare import circolari
try:
    # Importa i router esistenti
    from .auth import router as auth_router
    from .routes_tts import router as tts_router
    from .routes_questions import router as questions_router
    from .routes_first_prompt import router as first_prompt_router
    from .routes_transcribe import router as transcribe_router
    
    # Importazione diretta del router da routes_interview.py
    from .routes_interview import router as interview_router
    
    # Dizionario dei router disponibili
    routers = {
        'auth': auth_router,
        'tts': tts_router,
        'questions': questions_router,
        'interview': interview_router,
        'first_prompt': first_prompt_router,
        'transcribe': transcribe_router
    }
    
    # Esporta direttamente i router per facilitare l'importazione in main.py
    __all__ = ['auth_router', 'tts_router', 'questions_router', 'interview_router', 'first_prompt_router', 'transcribe_router', 'routers']
    
except ImportError as e:
    import sys
    print(f"Errore di importazione nei router API: {e}", file=sys.stderr)
    # Crea dizionario vuoto per evitare errori se l'importazione fallisce
    routers = {}