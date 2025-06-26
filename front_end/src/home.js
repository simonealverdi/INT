// ------------------------------------------------------------
// Config
// ------------------------------------------------------------
import { CONFIG, logDebug } from './config.js';
const BACKEND = CONFIG.BACKEND_URL;

// Variabile per memorizzare il token di autenticazione
let authToken = null;

document.addEventListener('DOMContentLoaded', () => {
  // --- Login Handling ---
  const loginForm = document.getElementById('loginForm');
  const loginOverlay = document.getElementById('loginOverlay');
  const mainContent = document.getElementById('mainContent');
  const loginError = document.getElementById('loginError');
  
  // Verifica se è già presente un token in localStorage
  const savedToken = localStorage.getItem('auth_token');
  if (savedToken) {
    // Valida il token con il backend
    validateToken(savedToken);
  }
  
  // Handler per il submit del form di login
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    try {
      // Mostra indicatore di caricamento
      loginError.textContent = 'Autenticazione in corso...';
      loginError.style.color = '#3498db';
      
      // Chiamata al backend per il login
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);
      
      const response = await fetch(`${BACKEND}/api/token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData,
        credentials: 'include'
      });
      
      const data = await response.json();
      
      if (response.ok && data.access_token) {
        // Login riuscito
        authToken = data.access_token;
        localStorage.setItem('auth_token', authToken);
        
        // Nascondi il popup e mostra il contenuto principale
        loginSuccess();
        
        // Dopo il login riuscito, verifica la sessione
        checkSession();
      } else {
        // Login fallito
        loginError.textContent = data.detail || 'Credenziali non valide';
        loginError.style.color = '#e74c3c';
      }
    } catch (error) {
      console.error('Login error:', error);
      loginError.textContent = 'Errore di connessione. Riprova più tardi.';
      loginError.style.color = '#e74c3c';
    }
  });
  
  /**
   * Genera un token di sviluppo per semplificare l'autenticazione durante la fase di sviluppo
   * Usa lo stesso formato già supportato in interview.js
   */
  function ensureDevToken() {
    // Se non c'è un token di autenticazione, crea un token fittizio per lo sviluppo
    if (!authToken) {
      console.log("Nessun token trovato, generazione token di sviluppo");
      const userId = "admin"; // Usa le credenziali semplificate preferite
      // Crea un token fittizio che il backend riconoscerà come valido per lo sviluppo
      const devToken = 'dev_token_' + btoa(userId) + '_' + Math.floor(Date.now() / 1000);
      
      // Salva nei vari storage per compatibilità con tutto il sistema
      localStorage.setItem('auth_token', devToken);
      sessionStorage.setItem('jwt_token', devToken); // Per compatibilità con interview.js
      authToken = devToken;
      
      console.log(`Token di sviluppo generato: ${devToken.substring(0, 20)}...`);
      return true;
    }
    return false;
  }
  
  /**
   * Verifica un token di autenticazione esistente
   */
  async function validateToken(token) {
    try {
      const response = await fetch(`${BACKEND}/api/check_session`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        credentials: 'include'
      });
      
      const data = await response.json();
      
      if (response.ok) {
        // Token valido, memorizza e mostra contenuto principale
        authToken = token;
        loginSuccess();
        return true;
      } else {
        // Token non valido o scaduto, rimuovi da localStorage e genera un token di sviluppo
        localStorage.removeItem('auth_token');
        const devTokenCreated = ensureDevToken();
        if (devTokenCreated) {
          loginSuccess();
          return true;
        }
        return false;
      }
    } catch (error) {
      console.error('Token validation error:', error);
      // In caso di errore, genera un token di sviluppo
      const devTokenCreated = ensureDevToken();
      if (devTokenCreated) {
        loginSuccess();
        return true;
      }
      return false;
    }
  }
  
  /**
   * Funzione chiamata quando il login è riuscito
   */
  function loginSuccess() {
    loginOverlay.style.display = 'none';
    mainContent.style.display = 'block';
  }
  
  // --- Main App Logic ---
  const fileInput = document.getElementById('questionFile');
  const fileNameDisplay = document.getElementById('fileName');
  const uploadButton = document.getElementById('uploadBtn');
  const uploadStatus = document.getElementById('uploadStatus');
  const startInterviewButton = document.getElementById('startInterview');
  // Aggiungi event listener al pulsante per l'avvio dell'intervista
  startInterviewButton.addEventListener('click', async function(e) {
    // Previene il comportamento predefinito
    e.preventDefault();
    
    // Verifica se il pulsante è disabilitato
    if (this.classList.contains('disabled')) {
      return;
    }
    
    // Mostra un messaggio di caricamento
    showStatus('Avvio dell\'intervista in corso...', 'loading');
    this.disabled = true;
    
    try {
      // Verifica autenticazione
      if (!authToken) {
        showStatus('Non sei autenticato. Ricarica la pagina e accedi.', 'error');
        this.disabled = false;
        return;
      }
    
    // Verifica il token di autenticazione
      console.log("Token prima della chiamata:", 
                 authToken ? authToken.substring(0, 20) + "..." : "MANCANTE");
                 
    // Chiamata all'endpoint per avviare l'intervista
      const response = await fetch(`${BACKEND}/api/interview/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        credentials: 'include'
      });
    
    // Processa la risposta
      const result = await response.json();
    
      if (response.ok && result.status === 'success') {
        // Salva l'ID dell'intervista nella sessionStorage
        if (result.interview_id) {
          console.log('Intervista avviata con ID:', result.interview_id);
          sessionStorage.setItem('interview_session_id', result.interview_id);
        }
      
        // Reindirizza alla pagina dell'intervista
        window.location.href = 'interview.html';
      } else {
        // Mostra un messaggio di errore
        showStatus(`Errore nell'avvio dell'intervista: ${result.message || 'Errore sconosciuto'}`, 'error');
        this.disabled = false;
      }
    } catch (error) {
      console.error('Errore durante l\'avvio dell\'intervista:', error);
      showStatus(`Errore di connessione: ${error.message}`, 'error');
      this.disabled = false;
    }
});

  // Verifica se esiste già una sessione valida
  checkSession();
  
  // Mostra il nome del file quando viene selezionato
  fileInput.addEventListener('change', function() {
    if (this.files && this.files[0]) {
      fileNameDisplay.textContent = this.files[0].name;
      uploadButton.disabled = false;
    } else {
      fileNameDisplay.textContent = 'Nessun file selezionato';
      uploadButton.disabled = false;
    }
  });
  
  // Carica le domande quando si fa clic sul pulsante
  uploadButton.addEventListener('click', function() {
    uploadQuestions();
  });
    
    /**
     * Upload questions to the backend
     */
    async function uploadQuestions() {
        if (!fileInput.files || !fileInput.files[0]) {
            showStatus('Seleziona un file prima di caricare.', 'error');
            return;
        }
        
        // Verifica se l'utente è autenticato
        if (!authToken) {
            showStatus('Non sei autenticato. Ricarica la pagina e accedi.', 'error');
            return;
        }
        
        // Show loading status
        showStatus('Caricamento in corso...', 'loading');
        uploadButton.disabled = true;
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        try {
            const response = await fetch(`${BACKEND}/api/questions/load`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${authToken}`
                },
                body: formData,
                credentials: 'include'
            });

            let result = null;
            let errorMsg = '';
            try {
                result = await response.json();
                
                // Una volta ricevuta la risposta dal server, cambia immediatamente il messaggio
                // Anche prima di verificare se ci sono errori, almeno sappiamo che i dati sono arrivati
                if (response.ok) {
                    showStatus('Domande caricate! Preparazione intervista...', 'success');
                }
            } catch (jsonErr) {
                errorMsg = `Risposta non valida dal server (status ${response.status})`;
            }

            if (response.ok && result && result.status === 'success') {
                // Salva il token dalla risposta JSON
                if (result.session_token) {
                    console.log('Salvando token dalla risposta JSON:', result.session_token);
                    sessionStorage.setItem('interview_token', result.session_token);
                }
                if (result.session_id) {
                    console.log('Salvando session_id dalla risposta:', result.session_id);
                    sessionStorage.setItem('interview_session_id', result.session_id);
                }
                
                // SEMPRE attiva il pulsante quando le domande sono caricate,
                // anche se i metadati sono ancora in elaborazione
                startInterviewButton.classList.remove('disabled');
                startInterviewButton.disabled = false; // Rimuovo anche l'attributo disabled
                console.log("Stato pulsante dopo caricamento:", 
                           startInterviewButton.classList.contains('disabled') ? "Ancora disabilitato" : "Abilitato",
                           "Attributo disabled:", startInterviewButton.disabled);
                // Verifica anche lo stile diretto
                console.log("Stile pulsante:", startInterviewButton.style.opacity, startInterviewButton.style.pointerEvents);
                
                // Controlla se i metadati sono in elaborazione
                if (result.metadata_processing && result.metadata_processing.in_progress) {
                    // Mostra un messaggio che indica che i metadati sono in elaborazione
                    showStatus(`${result.count} domande caricate con successo! Puoi iniziare l'intervista mentre i metadati vengono elaborati in background (${result.metadata_processing.processed_questions}/${result.metadata_processing.total_questions}).`, 'success');
                    
                    // Opzionale: avvia un polling per verificare lo stato dei metadati
                    startMetadataPolling();
                } else {
                    showStatus(`${result.count} domande caricate con successo!`, 'success');
                }
            } else {
                // Gestione errori dettagliata
                if (response.status === 401) {
                    showStatus('Non autorizzato: effettua di nuovo il login.', 'error');
                } else if (result && result.message) {
                    showStatus(`Errore: ${result.message}`, 'error');
                } else if (errorMsg) {
                    showStatus(errorMsg, 'error');
                } else {
                    showStatus(`Errore sconosciuto (status ${response.status})`, 'error');
                }
                uploadButton.disabled = false;
            }
        } catch (error) {
            showStatus('Errore di connessione al server. Riprova più tardi.', 'error');
            console.error('Upload error:', error);
            uploadButton.disabled = false;
        }
    }
    
    /**
     * Check if a valid session exists
     */
    async function checkSession() {
        try {
            // Aggiungi il token di autenticazione all'header
            const headers = authToken ? {
                'Authorization': `Bearer ${authToken}`
            } : {};
            
            const response = await fetch(`${BACKEND}/api/check_session`, {
                method: 'GET',
                headers: headers,
                credentials: 'include' // Important for cookies
            });
            
            const result = await response.json();
            
            if (result.valid && result.questions_loaded) {
                // Questions already loaded, enable interview button
                startInterviewButton.classList.remove('disabled');
                showStatus('Domande già caricate. Puoi iniziare l\'intervista.', 'success');
            }
        } catch (error) {
            console.error('Session check failed:', error);
        }
    }
    
    /**
     * Show status message with optional styling
     */
    function showStatus(message, type) {
        uploadStatus.textContent = message;
        uploadStatus.className = 'status-message';
        
        if (type) {
            uploadStatus.classList.add(type);
        }
    }
    
    /**
     * Inizia il polling dello stato dei metadati
     * Controlla periodicamente lo stato dell'elaborazione dei metadati e aggiorna l'interfaccia utente
     */
    let pollingInterval = null;
    function startMetadataPolling() {
        // Ferma eventuali polling precedenti
        if (pollingInterval) {
            clearInterval(pollingInterval);
        }
        
        // Avvia un nuovo polling ogni 2 secondi
        pollingInterval = setInterval(async () => {
            try {
                // Verifica lo stato dei metadati
                const response = await fetch(`${BACKEND}/api/questions/metadata-status`, {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${authToken}`
                    }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    
                    // Aggiorna il messaggio di stato con la percentuale di completamento
                    if (data.status === 'success' && data.metadata_processing) {
                        const meta = data.metadata_processing;
                        
                        if (!meta.in_progress) {
                            // Elaborazione completata
                            showStatus(`Metadati generati per tutte le ${meta.total_questions} domande. Pronto per iniziare l'intervista!`, 'success');
                            // Ferma il polling
                            clearInterval(pollingInterval);
                            pollingInterval = null;
                        } else {
                            // Ancora in elaborazione, mostra la percentuale
                            const percent = Math.round((meta.processed_questions / meta.total_questions) * 100);
                            showStatus(`Domande caricate! Metadati in elaborazione: ${percent}% completato (${meta.processed_questions}/${meta.total_questions}). Puoi comunque iniziare l'intervista.`, 'success');
                        }
                    }
                }
            } catch (error) {
                console.error('Errore durante il controllo dei metadati:', error);
            }
        }, 2000); // Controlla ogni 2 secondi
    }
});
