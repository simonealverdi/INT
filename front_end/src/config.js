/**
 * Configurazione centrale per l'applicazione
 * Questo file contiene impostazioni che potrebbero variare tra ambienti
 */

// Rilevamento automatico del backend
const detectBackendUrl = () => {
  // Se in produzione, usa lo stesso host ma con percorso /api
  if (window.location.hostname !== 'localhost' && 
      window.location.hostname !== '127.0.0.1') {
    // In produzione: stesso dominio ma con /api
    return `${window.location.protocol}//${window.location.hostname}/api`;
  }
  
  // In sviluppo, prova prima il backend standard
  const devBackends = [
    'http://127.0.0.1:8000',  // Backend standard
    'http://localhost:8000',  // Alternativa locale
  ];
  
  // Usa il primo backend disponibile
  return devBackends[0];
};

// Configurazione esportata
export const CONFIG = {
  // URL del backend con rilevamento automatico
  BACKEND_URL: detectBackendUrl(),
  
  // ID utente di default (usato solo per demo)
  DEFAULT_USER_ID: 'utente_demo',
  
  // Timeout per silenzio considerato fine turno (ms)
  SILENCE_TIMEOUT_MS: 2000,
  
  // Durata animazione predefinita (ms)
  DEFAULT_ANIMATION_DURATION: 300,
  
  // Flag debug
  DEBUG: true
};

// Funzione di logging condizionale
export const logDebug = (...args) => {
  if (CONFIG.DEBUG) {
    console.log('[DEBUG]', ...args);
  }
};

export default CONFIG;
