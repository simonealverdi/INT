import logging, os

# Determina il livello di logging dalla variabile d'ambiente o usa INFO come default
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_level_num = getattr(logging, log_level, logging.INFO)

class SensitiveDataFilter(logging.Filter):
    """Filtra messaggi di log che contengono dati sensibili o verbose."""
    def __init__(self, keywords=None):
        super().__init__()
        self.keywords = keywords or ["base64", "audio", "SSML", "TTS"]
    
    def filter(self, record):
        # Converti il record in stringa per l'analisi
        log_message = str(record.msg)
        
        # Riduce il livello di logging per messaggi con parole chiave sensibili
        # ma meno importanti (da INFO a DEBUG)
        if any(keyword in log_message for keyword in self.keywords):
            # Se il livello è INFO (20) e contiene una keyword sensibile, aumenta a DEBUG (10)
            # In questo modo i messaggi verranno mostrati solo se il livello è DEBUG
            if record.levelno == logging.INFO:
                record.levelno = logging.DEBUG
                record.levelname = "DEBUG"
        return True

# Utility per troncare stringhe lunghe nei log
def truncate_for_logging(text, max_length=100):
    if text is None:
        return None
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length] + '...'

def setup_logger(name: str = __name__):
    """Configura e restituisce un logger con le impostazioni standard"""
    logging.basicConfig(
        level=log_level_num,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name)
    logger.setLevel(log_level_num)
    logger.addFilter(SensitiveDataFilter())
    return logger

# Crea un'istanza del logger
logger = setup_logger(__name__)
