# Export della logica di business dell'applicazione

# Import dei servizi di business logic principali
try:
    from Main.application.user_session_service import get_state, has_active_session, reset_session, load_script, get_session_info, SESSIONS, SCRIPT
    from Main.application.interview_service import generate_question_structure, _process_next_step, handle_empty_transcription
except ImportError as e:
    import sys
    print(f"Errore di importazione nella logica applicativa: {e}", file=sys.stderr)
