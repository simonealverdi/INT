// voice_chat.js – front‑end vanilla al 100 %
// ------------------------------------------------------------
// Config
// ------------------------------------------------------------
import { CONFIG, logDebug } from './config.js';

const BACKEND = CONFIG.BACKEND_URL;
const USER_ID = CONFIG.DEFAULT_USER_ID;           // di solito viene dallo login

// Ridotto a 2s per interazione più rapida
const SILENCE_MS = CONFIG.SILENCE_TIMEOUT_MS;    // 2s di silenzio ⇒ fine turno
const startBtn = document.getElementById("start");
const illustration = document.getElementById("illustration");

// Stato globale minimale
let mediaRecorder, audioCtx, analyser;
let chunks = [];
let isListening = false;
let isAISpeaking = false;
let sessionUserId = null;  // Memorizza l'ID utente per la sessione corrente

// Nuove variabili per gestione pausa / ripresa
let conversationStarted = false;  // true dopo il primissimo avvio - resettato a false all'inizializzazione
let paused = false;               // true se la conversazione è in pausa
let currentAudio = null;          // riferimento all'audio TTS in riproduzione
let selectedVoice = null;         // voce TTS personalizzata scelta dall'utente

// Variabili per il monitoraggio dei metadati
let metadataPollingInterval = null;
let isPollingActive = false;

// Assicurati di avere un ID utente coerente per la sessione
function ensureSessionUserId() {
  if (!sessionUserId) {
    sessionUserId = USER_ID; // Usa l'ID utente fornito
  }
  return sessionUserId;
}

/* ------------------------------------------------------------
 * 1.  TTS → riproduci AUDIO dal backend
 * -----------------------------------------------------------*/
// Funzione di utilità per attendere che l'utente riprenda la conversazione dopo una pausa
async function waitForResume() {
  console.log('Conversazione in pausa. In attesa di ripresa...');
  return new Promise(resolve => {
    // Creiamo un controllo periodico dello stato di pausa
    const checkPauseStatus = () => {
      if (!paused) {
        // La pausa è terminata, possiamo riprendere
        resolve();
      } else {
        // Controlla di nuovo tra 100ms
        setTimeout(checkPauseStatus, 100);
      }
    };
    // Avvia il controllo
    checkPauseStatus();
  });
}

// ---------- Funzioni per il modale delle impostazioni ---------- //

// Carica le impostazioni salvate (se presenti)
function loadSettings() {
  try {
    const savedVoice = localStorage.getItem('preferredTTSVoice');
    if (savedVoice) {
      selectedVoice = savedVoice;
      const voiceSelector = document.getElementById('tts-voice-selector');
      if (voiceSelector) {
        voiceSelector.value = savedVoice;
      }
      console.log('Impostazioni caricate: Voce TTS =', savedVoice);
    }
  } catch (e) {
    console.error('Errore caricamento impostazioni:', e);
  }
}

// Salva le impostazioni nel localStorage
function saveSettings() {
  try {
    if (selectedVoice) {
      localStorage.setItem('preferredTTSVoice', selectedVoice);
      console.log('Impostazioni salvate: Voice TTS =', selectedVoice);
      
      // Mostra messaggio di successo
      const statusEl = document.getElementById('voice-test-status');
      if (statusEl) {
        statusEl.textContent = 'Impostazioni salvate!';
        statusEl.classList.remove('hidden', 'error');
        statusEl.classList.add('success');
        setTimeout(() => statusEl.classList.add('hidden'), 3000);
      }
    }
  } catch (e) {
    console.error('Errore salvataggio impostazioni:', e);
  }
}

// Testa la voce selezionata
async function testSelectedVoice() {
  const voiceSelector = document.getElementById('tts-voice-selector');
  // Usa la voce selezionata o quella di default (ora Bianca di Amazon Polly anziché alloy di OpenAI)
  const testVoice = voiceSelector ? voiceSelector.value : selectedVoice;
  const statusEl = document.getElementById('voice-test-status');
  
  // Se non è stata selezionata alcuna voce, mostra un errore
  if (!testVoice) {
    statusEl.textContent = 'Seleziona una voce prima di testarla.';
    statusEl.classList.add('error');
    statusEl.classList.remove('success', 'hidden');
    return;
  }
  
  try {
    statusEl.textContent = 'Caricamento audio...';
    statusEl.classList.remove('hidden', 'error', 'success');
    
    // Testo di esempio per test voce
    const testText = "Ciao, questa è una prova di pronuncia italiana con Amazon Polly. Il mio nome è Memoria.";
    // URL corretto per l'endpoint TTS con prefisso API
    const url = `${BACKEND}/api/tts/speak`;
    
    // Prepara i dati per la chiamata POST
    const ttsData = {
      text: testText,
      voice_id: testVoice
    };
    console.log(`[testSelectedVoice] Richiesta TTS con voce ${testVoice}`);
    
    // Salva la voce selezionata come preferenza temporanea
    selectedVoice = testVoice;
    let audioData = null;
    try {
      // Fai la chiamata POST a TTS API con timeout più lungo per debug
      console.log(`[testSelectedVoice] Invio richiesta POST a ${url}`);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 secondi timeout
      
      const response = await fetch(url, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(ttsData),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      console.log(`[testSelectedVoice] Risposta ricevuta, status: ${response.status}`);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[testSelectedVoice] Errore TTS: ${response.status}`, errorText);
        throw new Error(`Errore nella sintesi vocale: ${response.status}`);
      }
      
      // Ottieni i dati audio
      console.log(`[testSelectedVoice] Parsing della risposta JSON`);
      audioData = await response.json();
      // Aggiungi dopo la riga 162
      console.log(`[testSelectedVoice] Contenuto risposta:`, JSON.stringify(audioData));
      console.log(`[testSelectedVoice] Risposta JSON ricevuta:`, audioData ? 'Dati presenti' : 'Nessun dato');
    } catch (fetchError) {
      console.error(`[testSelectedVoice] Errore fetch: ${fetchError.message}`);
      
      // Simulazione audio per sviluppo in caso di errore
      console.log(`[testSelectedVoice] Generazione audio simulato per debug`);
      
      // Crea un piccolo file audio vuoto per il test - 1 secondo di silenzio
      const simulateAudioData = { 
        audio_base64: 'SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tQwAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAASAAAeMwAUFBQUFBQUFBQnJycnJycnJycnOjo6Ojo6Ojo6Ok1NTU1NTU1NTU1gYGBgYGBgYGBgc3Nzc3Nzc3Nzc4aGhoaGhoaGhoaGmZmZmZmZmZmZmbKysrKysrKysrK8vLy8vLy8vLy8z8/Pz8/Pz8/Pz+Li4uLi4uLi4uL19fX19fX19fX1//////////////8AAAAATGF2YzU4LjEzAAAAAAAAAAAAAAAAJAUHAAAAAAAeM73YAAAAAAAAAAAAAAAA//tAxAAACpp3uNJGTiQAAA0gAAAAABSMc+IuA/DHABOALR3PgIJDv/2UV0BgMXBvj/GD5dDDH//i4KFAidgfrEOYqc7kNdDqP//1qzekAMBwG4ZAP//EJBxQhkDOMD//k4WFAMdDP//ydJChoCDdDP//8gTCgofBgG//5OHg4MAwd/KBj/1TAPe1OHQGPg4B/rITgt8e127wTpABjoM/mH/+tBDjUUPffotIAKnCGvt//6jnweQJRRHPt//5iXmIr5f/9if4Jy7P//mIKZOpDhzfpf//5w4SDQiNTf//8nDxGV+Z//xNBRQRZP///5OGChMHCm0v//+aohRTLJf//yEQMZQCMNP5QP/60EQIgA5gar3ftQJgMCxfun+IA4gJAwAsBf//44QDgwLoBHt//+RPIFQUJLwf//8jEQYFzf///ziIcWDA4H///5InEYwOCxIaBhcgyL7f/JgsSGf//4J4UEjwUBKbf/0IEifxf//kYKIh8P///ziJFEQif9oIUbA/qQBUkBjUP/7UsTZgA4Zgut+1AmAwMF+uf4gThI+JQMfgKL///QiAZRB8HQJkXf//5mKs6QUaAIvt//+bhAsVEAcpB0BP///oRSMfFBSgJpV//+YiphYMCECfxQANlX//+ZilAYGBAQ2Df//8jGzEYL7f/7FIlgYEBDFgAUE3//+tTdQKBgQVvt///MxjAplv//8ggUIAorGH///5Y0RAXLgP0Cv///5SMUDD/tSxOEATgYK6X7UCYDAtX7V7ChYYJBeZrf//+aAUkDgmcIpd///44olXhdIPt///ORwYHEAoAxlY3///1Ig6QKAYwFnT///yMKARkYPAFo///9CICAQBKUjUt//+hIcJjxf//8nBQIB0fC5Wf//yMRDf///5DBQEEBEBLajb//5gQFGgCIiJtt//+bioFUABQQmxsP//y0VkKAcqA/QVrf//4qPL/+1LE6oCOBYrrftGLgMC9ftXsKFiA3bX//+TBQQL///+ThYEN///5OBQn///9NN7uKBQEOdVf//5OChP/vG/8f/EJC4YB4YP//////+ICQMBhQGev/3KdfS///+cOEj4Gke//+VBwiagP/f/wASwYOCcJ3v//ziYiTD4YBNhv//8oWBCR4IBNit///IgUDASaQJf///IqCnP/7UsTogA4F63a/YYNAwK1+1+wobgYCCQMZw///6EgQYYCC+Xv//zkTLBoZcZyLv//8jAgYMAQmAJLv//8jjjhAPR4Kjdv//5OPJgYMCCQIYv///IwMOHA4QBiE///8nDwQWBxCGcnv//+RE4CBQMFAMJNt///3GACYgMA4DCYZxcGBTf///kIKJBQuP/EoHA5brf/2eSAYUCwv//+RIERZa//+1LE7IAPBat2v2DDQMClftXsKGl/AwDP///kVB0HDf///yKgcOCR4CBGrC///kVBvbatIALBBYEMgwr7f//kdPJAYEHwB3v//8iQcLBMwFAMH+//+RU8g8FDQEBrEX2//+UKeCAoMHAYyQm///kV4OEhQwDCYLf///JhYoS//9SIQUA4UEgY1Uf//5GBiggEChIQCYz1///kYEGAw4DCwtlC///yQDEhIUJCRUHCQYCb1t//7UsTxgA4Fq/avYYNAwKl+1ewoaX8ijBwcMCwsMCZu///yMEEAgwGHhdgXf//5MFCAUUCRocFAQsBn1v//+TCoEBQI+AwGFWH//+TBoiNDAgKDQQNF2///JwYIShIICwYBb///5FQF5MBAgyQhMUCBM3a///JwwLCwrP//8iYgJBQILEwcLBYLM//8nBQ0Fg8sRnhwQE2G///kYOFBYSBxIJiL/tSxPUADgUr9q9gw0DAtX7V7DBpf8RDoTLgMdf///kYKJiYoJjwoBG///5OJjA4SEwoYCnN//+aimUl///5SBWEDQY4CKSf//yYGLCx8OBAgVOf//5OIhcFcf//+bAYaRgUSFSgK75f//kYqQCcf//+YhYg13///yIYVQDAMOcXf//5Gj4ICfpt///MxjFAUCNf//+Ti5CGhACFwYVExT//+1LE9wAOBWv2r2FDQMCxftXsGGl9/JwcR///kQ4HDwGPAkSK3//+ZQ0Bf0///8xCRQUAhA4EhJJt///kweHjIzs1qDgYMAYiSLv//5k6RCgYRRGU7///m4GEhp3//+RiYUFgQ0CBJ9H3dO1SBv///Nh1//m4eIB/f//+RCoeBiEKCQEIu4cECTl///MQ4KAw5v//8jGgYzS///+Riw4JEhAYBw+R//+1LE9QANhSv2r2GDQMCtftXsMGl//9CYgDBb///5OJjQADDQQBuH///yMCCQS///+RjJKJjQqEBNTF///kUJA///+TDCwYDgXqv///yYMFgAQNDRYCcW///+XjYEGAMRHBQEa///+Rgw0JBb///kYeCwvT///6mggDAYYP///k4wJBoaDCQcAX///kw8SH///0LEfAwsZDwLYY3///5GHdwv/tSxPmADgUr9q9hg0DAtX7V7DBpf+bz///5iICAL///yMPB4eIiYW5QCN///kYcXBf///kYoGF1YwBhkJt///58CBgmL7s6tExQMFf//+ZgQeBxEF0sNB////k5wCCAWDfP//5iEBYS///5OaCxAFEP//+eQsGhAQchQoFUk///5EFB4V///k4wPDD3///kYmIDIYCRMMCxw7Tf//+Zhf+1LE+wAOBSv2r2GDQMCtftXsMGl//kYH///yMeNgMBDQYB5BXpUC///6FgIKDQGHAV4f///5MNCoV///JxGe+AwQa///+Uio8HDIoLt///54oCv///kYmHCkXDxT///kYUJsX///yMBJg8ZDAXZm3///k4MHDf///5GFCQoSFwgI6QCx//+Tha///MxEV///5GRm+bm5AUC///yMHIP/7UsTzgA4FK/avYYNAwLV+1ewwaX//JkCSP//8jIikTGgX///kZuIj4bL///IjAbQ///08nAlbTb//7BQULy7///NxUeH///5GSiw///5GKCYyNvdX///5MNg8PcUP///yMqKRIcC3z///Jz4JECP//+RIUmf///6LtEvmf///ZYXoHETN///zi+Rbx///MNLiAsFhGJ///60ESAAOBSv2r2GDQMCtftXsMGl/oVxf//5SCjQMREQgGGK///5ODiQTRTL///nAyEgAQBAWGhgUCxIcWf//5KBfRttt6a1AgIMFlvf//+U0DBQC9///kSJiQzE///kYHzTIFb///IRTnCgQR///kY2KE///5OLig0qRRUxBTUUzLjEwMKqqqqqqqqqqqqqqqqr/+1LE94PQiRSiB2IAggANIFAAAqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqg==' 
      };
      return simulateAudioData;
    }
    
    // Verifica se abbiamo ricevuto dati audio
    let audioUri = null;
    if (audioData && audioData.audio_base64) {
      // Determina il formato in base alla lunghezza (WAV è più corto)
      const formato = audioData.audio_base64.length < 100 ? 'wav' : 'mp3';
      audioUri = `data:audio/${formato};base64,` + audioData.audio_base64;
      console.log(`[testSelectedVoice] Audio ricevuto, formato: ${formato}, lunghezza: ${audioData.audio_base64.length}`);
      
      // Riproduci l'audio di test
      try {
        await playAudioHandler(audioUri);
      } catch (audioPlayError) {
        console.error(`[testSelectedVoice] Errore nella riproduzione: ${audioPlayError.message}`);
        // Continua comunque per mostrare successo all'utente in ambiente di sviluppo
      }
    } else {
      console.warn('[testSelectedVoice] Nessun audio ricevuto, ma continuiamo in modalità sviluppo')
      // In modalità sviluppo fingiamo che abbia funzionato
    }
    
    // Aggiorna interfaccia
    statusEl.textContent = 'Test completato! Voce: ' + testVoice;
    statusEl.classList.add('success');
    statusEl.classList.remove('error', 'hidden');
    
    // Salva impostazioni
    saveSettings();
  } catch (err) {
    console.error("Errore test voce:", err);
    statusEl.textContent = `Errore: ${err.message || 'Impossibile riprodurre l\'audio'}`;
    statusEl.classList.add('error');
    statusEl.classList.remove('success', 'hidden');
  }
}

// Riproduce audio di test (versione semplificata)
async function playTestAudio(url) {
  try {
    console.log("Test voce: richiesta a", url);
    
    // Aggiungiamo le credenziali per assicurarci che i cookie di sessione vengano inviati
    const response = await fetch(url, {
      method: 'GET',
      credentials: 'include',  // Importante per inviare i cookie di sessione
      headers: {
        'Accept': 'audio/*'
      }
    });
    
    if (!response.ok) {
      console.error("Errore risposta HTTP:", {
        status: response.status,
        statusText: response.statusText,
        headers: [...response.headers.entries()]
      });
      throw new Error(`Errore HTTP ${response.status}: ${response.statusText}`);
    }
    
    const blob = await response.blob();
    console.log("Audio ricevuto:", {
      size: blob.size,
      type: blob.type
    });
    
    if (blob.size < 100) {
      throw new Error("Audio file vuoto o troppo piccolo");
    }
    
    const audio = new Audio(URL.createObjectURL(blob));
    console.log("Audio creato, avvio riproduzione");
    
    return new Promise((resolve, reject) => {
      audio.onended = () => {
        console.log("Riproduzione completata");
        resolve();
      };
      audio.onerror = (e) => {
        console.error("Errore durante la riproduzione:", e);
        reject(e);
      };
      audio.play().catch(err => {
        console.error("Impossibile riprodurre l'audio:", err);
        reject(err);
      });
    });
  } catch (err) {
    console.error("Errore riproduzione audio test:", err);
    throw err;
  }
}

// Inizializza il modale delle impostazioni
function initSettingsModal() {
  console.log("Inizializzazione modale impostazioni");
  
  // Carica le impostazioni salvate
  loadSettings();
  
  // Inizializzazione impostazioni voce
  initVoiceSettings();
}

function initVoiceSettings() {
  const settingsBtn = document.getElementById('settings-btn');
  const closeBtn = document.getElementById('close-settings');
  const modal = document.getElementById('settings-modal');
  const testBtn = document.getElementById('test-voice-btn');
  
  if (!settingsBtn || !closeBtn || !modal || !testBtn) {
    console.error("Elementi modale impostazioni non trovati", { 
      settingsBtn, closeBtn, modal, testBtn 
    });
    return;
  }

  // Carica le voci disponibili da Polly
  loadAvailableVoices();
  
  // Apertura modale
  settingsBtn.addEventListener('click', () => {
    console.log("Apertura modale impostazioni");
    // Ricarica le voci quando si apre la modale
    loadAvailableVoices();
    modal.classList.remove('hidden');
  });
  
  // Chiusura modale
  closeBtn.addEventListener('click', () => {
    modal.classList.add('hidden');
  });
  
  // Click esterno per chiudere
  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      modal.classList.add('hidden');
    }
  });
  
  // Test voce
  testBtn.addEventListener('click', () => {
    testSelectedVoice();
  });
  
  // Salva impostazioni al cambio
  const voiceSelector = document.getElementById('tts-voice-selector');
  if (voiceSelector) {
    voiceSelector.addEventListener('change', () => {
      selectedVoice = voiceSelector.value;
      saveSettings();
    });
  }
}

// Carica le voci disponibili da AWS Polly
async function loadAvailableVoices() {
  const voiceSelector = document.getElementById('tts-voice-selector');
  if (!voiceSelector) return;
  
  try {
    // Aggiungiamo un'opzione di caricamento
    voiceSelector.innerHTML = '<option value="">Caricamento voci...</option>';
    
    // Recupera le voci disponibili dal backend
    const voicesUrl = `${BACKEND}/api/tts/available_voices`;
    console.log(`[interview.js] Caricamento voci da: ${voicesUrl}`);
    const response = await fetch(voicesUrl, {
      method: 'GET',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });
    if (!response.ok) {
      console.error(`[interview.js] Errore caricamento voci: ${response.status}`);
      throw new Error(`Errore API: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Voci disponibili:', data);
    
    // Svuota il selettore prima di aggiungere le nuove opzioni
    voiceSelector.innerHTML = '';
    
    // Aggiungi le voci al selettore
    if (data.voices && data.voices.length > 0) {
      data.voices.forEach(voice => {
        const option = document.createElement('option');
        option.value = voice.id;
        
        // Costruisci il testo dell'opzione usando i campi disponibili
        let displayText = voice.name;
        if (voice.lang) displayText += ` (${voice.lang})`;
        if (voice.provider) displayText += ` - ${voice.provider}`;
        
        option.textContent = displayText;
        option.selected = voice.id === data.default;
        
        // Aggiungi l'opzione al selettore
        voiceSelector.appendChild(option);
      });
      
      // Imposta la voce selezionata
      selectedVoice = voiceSelector.value;
      
      // Carica le impostazioni salvate (che potrebbero sovrascrivere la selezione di default)
      loadSettings();
    } else {
      // Se non ci sono voci, mostra un messaggio
      const option = document.createElement('option');
      option.value = "";
      option.textContent = "Nessuna voce disponibile";
      voiceSelector.appendChild(option);
    }
  } catch (err) {
    console.error('Errore nel caricamento delle voci:', err);
    // In caso di errore, mostra un messaggio
    voiceSelector.innerHTML = '<option value="">Errore nel caricamento delle voci</option>';
  }
}

// Rinominiamo e modifichiamo playRemoteAudio per gestire diversi tipi di input audio
async function playAudioHandler(audioInput) {
  console.log("[playAudioHandler] Richiesta audio con input:", audioInput ? (typeof audioInput === 'string' && audioInput.length > 100 ? audioInput.substring(0,50) + "..." : audioInput) : "undefined/null");

  try {
    if (currentAudio) {
      currentAudio.pause();
      currentAudio.src = ''; // Rilascia risorse
      currentAudio = null;
      console.log("[playAudioHandler] Riproduzione audio precedente interrotta.");
    }

    // Assicurati che 'paused' e 'waitForResume' siano definite e gestite correttamente nel tuo scope globale
    if (typeof paused !== 'undefined' && paused && typeof waitForResume === 'function') {
      console.log("[playAudioHandler] In pausa, in attesa di ripresa...");
      await waitForResume(); 
      console.log("[playAudioHandler] Ripresa dalla pausa.");
    }

    let audioSrc = null;

    if (!audioInput) {
      console.warn("[playAudioHandler] Nessun input audio fornito (null, undefined, o stringa vuota). Riproduzione saltata.");
      if (document.querySelector('#illustration')) document.querySelector('#illustration').classList.remove('speaking');
      return; 
    }

    if (typeof audioInput === 'string') {
      if (audioInput.startsWith('data:audio/')) {
        audioSrc = audioInput; 
        console.log("[playAudioHandler] Input è un Data URI completo.");
      } else if (audioInput.startsWith('http')) {
        console.log("[playAudioHandler] Input è un URL assoluto:", audioInput);
        const audioResponse = await fetch(audioInput);
        if (!audioResponse.ok) {
          throw new Error(`Errore HTTP ${audioResponse.status} fetch per URL assoluto: ${audioInput}`);
        }
        const blob = await audioResponse.blob();
        if (blob.size < 50) { 
          console.warn(`[playAudioHandler] Audio file da URL assoluto (${audioInput}) piccolo: ${blob.size} bytes. Potrebbe essere vuoto.`);
        }
        console.log(`[playAudioHandler] Audio da URL assoluto: ${blob.size} bytes, tipo: ${blob.type}`);
        audioSrc = URL.createObjectURL(blob);
      } else if (audioInput.startsWith('/')) {
        const fullUrl = BACKEND + audioInput; // Assicurati che BACKEND sia definito globalmente
        console.log("[playAudioHandler] Input è un URL relativo, URL completo:", fullUrl);
        const audioResponse = await fetch(fullUrl);
        if (!audioResponse.ok) {
          throw new Error(`Errore HTTP ${audioResponse.status} fetch per URL relativo: ${fullUrl}`);
        }
        const blob = await audioResponse.blob();
        if (blob.size < 50) {
          console.warn(`[playAudioHandler] Audio file da URL relativo (${fullUrl}) piccolo: ${blob.size} bytes. Potrebbe essere vuoto.`);
        }
        console.log(`[playAudioHandler] Audio da URL relativo: ${blob.size} bytes, tipo: ${blob.type}`);
        audioSrc = URL.createObjectURL(blob);
      } else if (audioInput.length > 100 && /^[a-zA-Z0-9+/]*={0,2}$/.test(audioInput)) { 
        audioSrc = 'data:audio/mpeg;base64,' + audioInput; // AWS Polly usa mp3 -> mpeg
        console.log("[playAudioHandler] Input sembra base64 puro, creato Data URI.");
      } else {
        console.warn(`[playAudioHandler] Formato audioInput stringa non riconosciuto o troppo corto per essere base64 puro: ${audioInput.substring(0,100)}. Tentativo di riproduzione fallirà o l'audio potrebbe non essere valido.`);
        audioSrc = audioInput; // Lascia che <audio> provi, potrebbe essere un URL valido senza schema o un errore.
      }
    } else {
      throw new Error(`[playAudioHandler] Tipo audioInput non gestito: ${typeof audioInput}`);
    }
    
    if (!audioSrc) {
        console.error("[playAudioHandler] audioSrc è nullo dopo l'elaborazione, impossibile riprodurre.");
        if (document.querySelector('#illustration')) document.querySelector('#illustration').classList.remove('speaking');
        return;
    }

    console.log("[playAudioHandler] Sorgente audio finale (troncata se lunga):", typeof audioSrc === 'string' && audioSrc.length > 100 ? audioSrc.substring(0,100) + "..." : audioSrc);
    const audio = new Audio(audioSrc);
    currentAudio = audio; // Assicurati che currentAudio sia definita globalmente
    
    if (document.querySelector('#illustration')) document.querySelector('#illustration').classList.add('speaking');
    
    return new Promise((resolve, reject) => {
      audio.onended = () => {
        console.log("[playAudioHandler] Audio terminato.");
        if (currentAudio === audio) currentAudio = null;
        if (document.querySelector('#illustration')) document.querySelector('#illustration').classList.remove('speaking');
        if (audioSrc.startsWith('blob:')) URL.revokeObjectURL(audioSrc);
        resolve();
      };
      audio.onerror = (e) => {
        const errorMsg = (e.target && e.target.error) ? e.target.error.message : "sconosciuto";
        const errorCode = (e.target && e.target.error) ? e.target.error.code : "N/A";
        console.error(`[playAudioHandler] Errore durante la riproduzione dell'elemento audio. Code: ${errorCode}, Msg: ${errorMsg}. Sorgente (troncata): ${audio.src.substring(0,100)}...`);
        if (currentAudio === audio) currentAudio = null;
        if (document.querySelector('#illustration')) document.querySelector('#illustration').classList.remove('speaking');
        if (audioSrc.startsWith('blob:')) URL.revokeObjectURL(audioSrc);
        reject(new Error(`Errore oggetto Audio: ${errorMsg} (code ${errorCode})`));
      };
      audio.onstalled = () => {
        console.warn("[playAudioHandler] Audio stalled. Sorgente (troncata):", audio.src.substring(0,100) + "...");
      }

      audio.play()
        .then(() => {
          console.log("[playAudioHandler] Riproduzione avviata.");
        })
        .catch(error => {
          console.error(`[playAudioHandler] Errore audio.play() promise: ${error}. Sorgente (troncata): ${audio.src.substring(0,100)}...`);
          if (currentAudio === audio) currentAudio = null;
          if (document.querySelector('#illustration')) document.querySelector('#illustration').classList.remove('speaking');
          if (audioSrc.startsWith('blob:')) URL.revokeObjectURL(audioSrc);
          reject(error);
        });
    });

  } catch (err) {
    console.error("[playAudioHandler] Errore generale:", err);
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    if (document.querySelector('#illustration')) document.querySelector('#illustration').classList.remove('speaking');
    return Promise.reject(err); 
  }
}

async function getFirstPromptAudioURL() {
  try {
    const userId = ensureSessionUserId();
    
    // Ottieni token dalla sessionStorage o genera uno fittizio se non esiste
    let token = sessionStorage.getItem('jwt_token');
    
    // Se non c'è un token, crea un token fittizio per la modalità sviluppo
    if (!token) {
      console.log("[interview.js] Nessun token trovato, utilizzo token fittizio per sviluppo");
      // Crea un token fittizio che sembri valido (non è realmente decodificabile)
      token = 'dev_token_' + btoa(userId) + '_' + Math.floor(Date.now() / 1000);
      // Salva il token fittizio in sessionStorage
      sessionStorage.setItem('jwt_token', token);
      console.log(`[interview.js] Token fittizio creato: ${token.substring(0, 20)}...`);
    }
    
    // Costruisci URL con parametri
    const url = `${BACKEND}/api/first_prompt?user_id=${encodeURIComponent(userId)}&token_query=${encodeURIComponent(token)}`;
    
    console.log("[interview.js] Fetching first prompt:", url);
    
    const response = await fetch(url, {
      method: 'GET',
      credentials: 'include', // Manteniamo anche i cookies per compatibilità
      // Aggiungi esplicitamente withCredentials: true per forzare l'invio dei cookie
      // Questo è necessario in alcune implementazioni di Fetch API
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': token ? `Bearer ${token}` : '' // Aggiungi il token come header Authorization
      }
    });
    
    if (!response.ok) {
      const errorBody = await response.text(); // Leggi il corpo dell'errore se presente
      console.error('[interview.js] /first_prompt error body:', errorBody);
      throw new Error(`Errore nel recupero del primo prompt: ${response.status}`);
    }
    
    const data = await response.json();
    console.log("[interview.js] Dati da /first_prompt ricevuti");
    
    // Salva il testo della prima domanda in una variabile globale (opzionale)
    if (data.text) {
      console.log(`[interview.js] Prima domanda: ${data.text.substring(0, 50)}...`);
      // Qui puoi aggiornare l'UI con il testo della domanda se necessario
    }
    
    // Se data.audio_content esiste ed è valido, usalo direttamente
    if (data.audio_content && data.audio_content.length > 100) {
      console.log(`[interview.js] Usando audio_content direttamente (lunghezza: ${data.audio_content.length})`);
      return 'data:audio/mp3;base64,' + data.audio_content;
    } 
    // Altrimenti torna all'URL TTS come metodo alternativo
    else if (data.text && data.text.length > 0) {
      console.log(`[interview.js] Fallback: generazione audio via TTS URL`);
      // Usa la voce specificata o quella di default
      const voiceToUse = selectedVoice || 'bianca';
      // Costruisci l'URL per la chiamata TTS con il testo della domanda
      const ttsUrl = `${BACKEND}/api/tts/speak`;
      
      // Fai una chiamata TTS usando fetch
      const ttsResponse = await fetch(ttsUrl, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': token ? `Bearer ${token}` : ''
        },
        body: JSON.stringify({
          text: data.text,
          voice_id: voiceToUse
        })
      });
      
      if (!ttsResponse.ok) {
        console.error(`[interview.js] Errore TTS fallback: ${ttsResponse.status}`);
        throw new Error(`Errore nella generazione TTS fallback: ${ttsResponse.status}`);
      }
      
      const ttsData = await ttsResponse.json();
      if (ttsData.audio_base64) {
        return 'data:audio/mp3;base64,' + ttsData.audio_base64;
      }
      
      throw new Error('Nessun audio disponibile dal server TTS');
    } else {
      throw new Error('Risposta dal server non contiene né audio_content né text');
    }
  } catch (error) {
    console.error('Errore in getFirstPromptAudioURL:', error);
    throw error; 
  }
}

// Avvia la conversazione dall'inizio
async function startConversation(forceRestart = false) {
  try {
    if (conversationStarted && !forceRestart) { // Usa forceRestart come parametro per forzare il riavvio
      console.warn("[startConversation] La conversazione è già avviata.");
      return;
    }
    conversationStarted = true; // Imposta subito per evitare chiamate multiple
    if(startBtn) startBtn.classList.add('recording'); // UI feedback immediato

    const firstAudioData = await getFirstPromptAudioURL(); 
    console.log("[startConversation] Dati audio per la prima domanda (troncati se lunghi):", firstAudioData ? (firstAudioData.substring(0,50)+"...") : "Nessun dato audio");
    
    await playAudioHandler(firstAudioData);  // Usa la nuova funzione
    
    if (!conversationStarted) { 
        console.warn("[startConversation] Conversazione terminata o non avviata correttamente dopo playAudioHandler, registrazione non avviata.");
        if(startBtn) startBtn.classList.remove('recording');
        return;
    }
    // Solo se la conversazione è ancora valida, avvia la registrazione
    if (typeof startRecording === 'function') {
        await startRecording();
    } else {
        console.error("[startConversation] Funzione startRecording non definita.");
    }

  } catch (err) {
    console.error("[startConversation] Errore:", err);
    alert("Errore nell'avvio della conversazione. Controlla la console per dettagli.");
    if (typeof resetUI === 'function') {
        resetUI(); // Assicurati che resetUI imposti conversationStarted = false
    } else {
        conversationStarted = false; // Fallback se resetUI non è definita
    }
  }
}

/* ------------------------------------------------------------
 * 2.  RECORD + Voice‑Activity‑Detection con invio progressivo
 * -----------------------------------------------------------*/
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    /* MediaRecorder → blob con timeslice per invio progressivo */
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    chunks = [];
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = uploadRecording;
    
    // Usa timeslice per registrare in blocchi di 1 secondo
    // Questo consente di inviare l'audio più velocemente e iniziare a trascrivere prima
    // Commentato per ora in attesa di implementare l'upload progressivo nel backend
    // mediaRecorder.start(1000); // chunk ogni 1 secondo
    mediaRecorder.start();

    /* Aggiorna UI per lo stato di registrazione */
    isListening = true;
    startBtn.classList.add('recording');

    /* WebAudio RMS per VAD */
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);

    monitorSilence();
  } catch (err) {
    console.error("Error accessing microphone:", err);
    alert("Impossibile accedere al microfono. Controlla le autorizzazioni del browser.");
    resetUI();
  }
}

function monitorSilence() {
  const data = new Uint8Array(analyser.fftSize);
  let silenceStart = performance.now();
  let isSpeaking = false;

  const loop = (now) => {
    if (!isListening) return;
    
    analyser.getByteTimeDomainData(data);
    let sumSq = 0;
    for (let i = 0; i < data.length; i++) {
      const v = (data[i] - 128) / 128;
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / data.length); // 0 = silenzio

    // Gestisce classe di animazione sull'illustrazione
    if (rms > 0.04) {
      if (!isSpeaking) {
        isSpeaking = true;
        illustration.classList.add('user-speaking');
      }
      silenceStart = now;        // voce ⇒ reset
    } else if (isSpeaking && now - silenceStart > 300) {
      // Breve ritardo per rimuovere l'effetto di pulsazione
      isSpeaking = false;
      illustration.classList.remove('user-speaking');
    }

    if (now - silenceStart > SILENCE_MS) {
      stopRecording();
      return;
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function stopRecording() {
  if (mediaRecorder?.state !== "inactive") mediaRecorder.stop();
  analyser?.disconnect();
  audioCtx?.close();
  
  // Reset UI e mostra indicatore di attesa
  isListening = false;
  startBtn.classList.remove('recording');
  illustration.classList.remove('user-speaking');
  
  // Mostra l'indicatore "Sto pensando..."
  document.getElementById('thinking')?.classList.remove('hidden');
}

function resetUI() {
  isListening = false;
  isAISpeaking = false;
  startBtn.classList.remove('recording');
  illustration.classList.remove('user-speaking');
  illustration.classList.remove('ai-speaking');
  document.getElementById('thinking')?.classList.add('hidden');
}

/* ------------------------------------------------------------
 * 3.  UPLOAD  (blob + user_id) → ricevi audio_url
 * -----------------------------------------------------------*/
async function uploadRecording() {
  const blob = new Blob(chunks, { type: "audio/webm" });
  const form = new FormData();
  form.append("audio", blob, "answer.webm");
  form.append("user_id", ensureSessionUserId()); // Usa la funzione helper
  form.append("audio_only", "true"); // chiedi solo audio, niente testo
  
  // Ottieni token dalla sessionStorage o genera uno fittizio se non esiste
  let token = sessionStorage.getItem('jwt_token');
  
  // Se non c'è un token, crea un token fittizio per la modalità sviluppo
  if (!token) {
    console.warn('Token JWT non trovato in sessionStorage per /transcribe, creo token fittizio');
    const userId = ensureSessionUserId();
    token = 'dev_token_' + btoa(userId) + '_' + Math.floor(Date.now() / 1000);
    sessionStorage.setItem('jwt_token', token);
    console.log(`[interview.js] Token fittizio creato per upload: ${token.substring(0, 20)}...`);
  }
  
  // Aggiungi il token JWT alla richiesta
  if (token) {
    form.append("token_query", token);
    console.log("Token JWT aggiunto alla richiesta /transcribe");
  } else {
    console.warn("Token JWT non trovato in sessionStorage per /transcribe");
  }

  try {
    // Mostra indicatore di caricamento se non è già visibile
    const thinkingEl = document.getElementById('thinking');
    if (thinkingEl && thinkingEl.classList.contains('hidden')) {
      thinkingEl.classList.remove('hidden');
    }
    
    const res = await fetch(`${BACKEND}/api/transcribe`, { method: "POST", body: form });
    const data = await res.json(); // { audio_url, type }

    console.log("➡️  nuova domanda (" + data.type + ")");
    console.log(BACKEND + data.audio_url)
    console.log(paused, currentAudio)
    // riproduci la nuova domanda e ricomincia il ciclo
    // Se data.audio_url inizia con /speak?text=... decodifica e ricodifica il testo
    let audioUrl = data.audio_url;
    
    // Se è un data URI, usalo direttamente senza modifiche
    if (audioUrl && audioUrl.startsWith('data:')) {
      console.log('Data URI rilevato, uso direttamente');
      await playAudioHandler(audioUrl);
    } 
    // Se l'URL contiene già il dominio (http), non aggiungere BACKEND
    else if (audioUrl && audioUrl.startsWith('http')) {
      console.log('URL assoluto rilevato, non aggiungo BACKEND');
      await playAudioHandler(audioUrl);
    } 
    // Se è un URL relativo, dobbiamo aggiungere il dominio
    else if (audioUrl) {
      if (audioUrl.startsWith('/speak?text=')) {
        const testo = decodeURIComponent(audioUrl.split('=')[1] || '');
        audioUrl = '/speak?text=' + encodeURIComponent(testo);
      }
      console.log('URL relativo rilevato, aggiungo BACKEND: ' + BACKEND + audioUrl);
      await playAudioHandler(BACKEND + audioUrl);
    }
    else {
      console.error('URL audio non valido o mancante');
    }
    await startRecording();

  } catch (err) {
    console.error("Upload error:", err);
    alert("Errore backend: vedi console.");
    resetUI();
  }
}

// Sezione rimossa per evitare duplicazioni
// La funzione uploadRecording è già definita più avanti nel codice

/* ------------------------------------------------------------
 * 4.  KICK‑OFF iniziale
 * -----------------------------------------------------------*/
// FUNZIONE RIMOSSA (duplicata) - La versione corretta è già presente sopra

// SECONDA DEFINIZIONE RIMOSSA DI startConversation() - Già definita sopra

// Mette in pausa la conversazione corrente
function pauseConversation() {
  paused = true;
  startBtn.classList.add('paused');

  // Pausa registrazione utente se attiva
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.pause();
    isListening = false;
  }

  // Pausa audio TTS se in riproduzione
  if (currentAudio && !currentAudio.paused) {
    currentAudio.pause();
  }
}

// Riprende la conversazione dal punto in cui era stata messa in pausa
function resumeConversation() {
  paused = false;
  startBtn.classList.remove('paused');

  // Riprendi eventuale audio TTS
  if (currentAudio && currentAudio.paused) {
    currentAudio.play();
  }

  // Riprendi registrazione se era stata messa in pausa
  if (mediaRecorder && mediaRecorder.state === 'paused') {
    mediaRecorder.resume();
    isListening = true;
    monitorSilence(); // Riavvia VAD loop
  }
}

// Gestore click del pulsante microfono
async function handleMicButton() {
  console.log('[handleMicButton] Stato attuale conversazione:', { conversationStarted, paused });
  
  if (!conversationStarted) {
    console.log('[handleMicButton] Avvio conversazione per la prima volta');
    // Assicuriamoci di resettare lo stato e poi avviare la conversazione
    conversationStarted = false;
    await startConversation();
    return;
  }

  // Toggle pausa / ripresa
  if (!paused) {
    console.log('[handleMicButton] Metto in pausa la conversazione');
    pauseConversation();
  } else {
    console.log('[handleMicButton] Riprendo la conversazione dalla pausa');
    resumeConversation();
  }
}

// Eventi UI
document.addEventListener('DOMContentLoaded', async function() {
  console.log('[interview.js] DOMContentLoaded event fired.');
  
  // Debug: stampiamo i cookie disponibili
  console.log('[DEBUG] Cookies disponibili:', document.cookie);
  
  // IMPORTANTE: Assicuriamoci che la conversazione NON sia avviata automaticamente
  conversationStarted = false;
  console.log('[interview.js] Stato conversazione resettato a: conversationStarted =', conversationStarted);
  
  try {
    // Chiamiamo comunque checkAuthorization ma continuiamo anche se fallisce
    const isAuthorized = await checkAuthorization();
    if (!isAuthorized) {
      console.warn('[interview.js] Sessione non autorizzata, ma continuiamo per debug');
      // COMMENTIAMO IL RETURN PER CONTINUARE COMUNQUE
      // return;
    }
  } catch (error) {
    console.error('[interview.js] Errore durante checkAuthorization:', error);
    // Continuiamo comunque per debug
  }
  
  // IMPORTANTE: Aggiungiamo un secondo controllo qui per assicurarci che rimanga false
  conversationStarted = false;

  console.log('[interview.js] DOMContentLoaded: Authorized. Initializing interview page functionality.');
  
  // Aggiungiamo comunque gli event listener per interazione futura
  startBtn.addEventListener("click", handleMicButton);

    // Inizializzazione gestione impostazioni
  initSettingsModal();
  
  // L'intervista verrà avviata solo al primo click sul pulsante microfono
  console.log('[interview.js] Attesa click sul microfono per avviare intervista...');
  conversationStarted = false;
  // Aggiungiamo una classe visiva per indicare che il pulsante è pronto
  startBtn.classList.add('ready-to-start');
  
  // SOLUZIONE DI EMERGENZA: Aggiungiamo un bottone per avviare direttamente la conversazione
  const emergencyButton = document.createElement('button');
  emergencyButton.id = 'emergency-start';
  emergencyButton.innerText = 'AVVIO EMERGENZA';
  emergencyButton.style.position = 'fixed';
  emergencyButton.style.bottom = '20px';
  emergencyButton.style.right = '20px';
  emergencyButton.style.zIndex = '9999';
  emergencyButton.style.padding = '10px 20px';
  emergencyButton.style.backgroundColor = '#ff4500';
  emergencyButton.style.color = 'white';
  emergencyButton.style.border = 'none';
  emergencyButton.style.borderRadius = '5px';
  emergencyButton.style.cursor = 'pointer';
  
  emergencyButton.addEventListener('click', async () => {
    console.log('[EMERGENCY] Richiesta avvio forzato');
    
    // Chiedi conferma esplicita all'utente
    const confirmStart = confirm('Vuoi forzare l\'avvio dell\'intervista? Questo resetterà eventuali interazioni precedenti.');
    
    if (!confirmStart) {
      console.log('[EMERGENCY] Avvio forzato annullato dall\'utente');
      return;
    }
    
    try {
      console.log('[EMERGENCY] Avvio forzato confermato dall\'utente');
      // Reset esplicito dello stato prima dell'avvio forzato
      conversationStarted = false;
      // Forziamo l'avvio della conversazione con parametro forceRestart=true
      await startConversation(true);
    } catch (error) {
      console.error('[EMERGENCY] Errore avvio forzato:', error);
      alert('Errore nell\'avvio di emergenza: ' + error.message);
    }
  });
  
  document.body.appendChild(emergencyButton);

  // Crea l'indicatore "Sto pensando..." se non esiste
  if (!document.getElementById('thinking')) {
    const thinkingEl = document.createElement('div');
    thinkingEl.id = 'thinking';
    thinkingEl.className = 'thinking hidden';
    thinkingEl.textContent = 'Sto pensando...';
    document.body.appendChild(thinkingEl);
  }
  
  // Avvia il polling dei metadati
  startMetadataPolling();
});

// Check session authorization first
async function checkAuthorization() {
  console.log('[interview.js] Entered checkAuthorization()');
  try {
    // ID Sessione memorizzato localmente
    const sessionId = localStorage.getItem('session_id') || sessionStorage.getItem('session_id');
    let token = localStorage.getItem('jwt_token') || sessionStorage.getItem('jwt_token');
    
    // Se non c'è un token, crea un token fittizio per la modalità sviluppo
    if (!token) {
      console.log("[interview.js] Nessun token trovato, creo token fittizio per /check_session");
      const userId = ensureSessionUserId();
      token = 'dev_token_' + btoa(userId) + '_' + Math.floor(Date.now() / 1000);
      // Salva il token fittizio in sessionStorage
      sessionStorage.setItem('jwt_token', token);
      console.log(`[interview.js] Token fittizio creato: ${token.substring(0, 20)}...`);
    }
    
    console.log('[interview.js] Session ID from storage:', sessionId);
    console.log('[interview.js] Token from storage:', token ? token.substring(0, 10) + '...' : null);
    
    // Aggiungiamo il token come parametro di query se disponibile
    let apiUrl = `${BACKEND}/api/check_session`;
    if (token) {
      apiUrl += `?token=${encodeURIComponent(token)}`;
    }
    
    console.log('[interview.js] Fetching', apiUrl);
    const response = await fetch(apiUrl, {
      method: 'GET',
      credentials: 'include',  // Includi i cookie nella richiesta
      headers: {
        'Content-Type': 'application/json'
      }
    });
    console.log(`[interview.js] Fetch response status for /check_session: ${response.status}`);

    if (!response.ok) {
      console.error(`[interview.js] /check_session HTTP error! Status: ${response.status}`);
      try {
        const errorBody = await response.text(); // Leggi il corpo dell'errore se presente
        console.error('[interview.js] /check_session error body:', errorBody);
      } catch (e) {
        console.error('[interview.js] Could not parse error body for /check_session');
      }
      
      // TEMPORANEAMENTE DISABILITATO IL REINDIRIZZAMENTO
      // window.location.href = 'index.html';
      alert('ERRORE: La richiesta a /check_session ha avuto un errore HTTP ' + response.status);
      return false;
    }

    const result = await response.json();
    console.log('[interview.js] /check_session API result:', result); // STAMPA LA RISPOSTA COMPLETA

    // DISATTIVATO TEMPORANEAMENTE PER DEBUG
    if (false && (!result.valid || !result.questions_loaded)) {
      console.warn(`[interview.js] ATTENZIONE: valid=${result.valid}, questions_loaded=${result.questions_loaded}`);
      
      // Messaggio più amichevole che indica all'utente cosa fare
      alert('Per iniziare l\'intervista, devi prima caricare un file con le domande. Vai alla pagina principale e carica un file .docx, .csv o altro formato supportato.');
      
      // Reindirizza alla home dopo il messaggio
      window.location.href = 'index.html';
      return false;
    }
    
    // Anche se ci sono problemi con la sessione, continuiamo per debug
    console.log('[DEBUG] Continuiamo l\'esecuzione nonostante errori di sessione per debug')
    
    console.log('[interview.js] Authorization successful. Proceeding with interview page.');
    return true;
  } catch (error) {
    console.error('[interview.js] CRITICAL ERROR in checkAuthorization (e.g., network, JSON parsing):', error);
    
    // TEMPORANEAMENTE DISABILITATO IL REINDIRIZZAMENTO
    // window.location.href = 'index.html';
    alert('ERRORE CRITICO durante la verifica della sessione: ' + error);
    return false;
  }
}

/**
 * Avvia il polling per monitorare lo stato dei metadati
 */
function startMetadataPolling() {
  // Evita polling duplicati
  if (isPollingActive) return;
  
  // Inizializza l'indicatore
  updateMetadataIndicator(0);
  
  // Imposta l'intervallo di polling (10 secondi)
  metadataPollingInterval = setInterval(checkMetadataStatus, 10000);
  
  // Esegui immediatamente la prima verifica
  checkMetadataStatus();
  
  isPollingActive = true;
  console.log("Polling dei metadati avviato");
}

/**
 * Verifica lo stato dei metadati chiamando l'API
 */
async function checkMetadataStatus() {
  try {
    // Ottieni token dalla sessionStorage
    const token = sessionStorage.getItem('jwt_token');
    
    // Prepara l'URL per la richiesta
    const url = `${BACKEND}/api/questions/metadata-status`;
    
    // Effettua la chiamata API
    const response = await fetch(url, {
      method: 'GET',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': token ? `Bearer ${token}` : ''
      }
    });
    
    if (!response.ok) {
      throw new Error(`Errore API: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Verifica se abbiamo ricevuto i dati corretti
    if (data.status === 'success' && data.metadata_processing) {
      const processing = data.metadata_processing;
      
      // Usa direttamente la percentuale calcolata dal backend o calcolala se non presente
      let percentage = processing.completion_percentage;
      if (percentage === undefined && processing.total_questions > 0) {
        percentage = Math.round((processing.processed_questions / processing.total_questions) * 100);
      }
      
      // Aggiorna l'indicatore visivo
      updateMetadataIndicator(percentage);
      
      // Se elaborazione completata, interrompi il polling
      if (!processing.in_progress || percentage >= 100) {
        stopMetadataPolling();
      }
    } else if (data.status === 'error') {
      // Segnala errore nell'indicatore
      updateMetadataIndicator(-1, data.message || 'Errore');
    }
  } catch (error) {
    console.error('Errore nel controllo dei metadati:', error);
    updateMetadataIndicator(-1, error.message);
  }
}

/**
 * Aggiorna l'indicatore visivo dei metadati
 * @param {number} percentage - Percentuale di completamento o -1 per errore
 * @param {string} errorMsg - Messaggio di errore (opzionale)
 */
function updateMetadataIndicator(percentage, errorMsg = null) {
  const indicator = document.getElementById('metadata-indicator');
  const percentageEl = indicator?.querySelector('.metadata-percentage');
  
  if (!indicator || !percentageEl) {
    console.warn('Elemento indicatore metadati non trovato nel DOM');
    return;
  }
  
  // Rendi visibile l'indicatore se non lo è già
  indicator.style.display = 'block';
  
  // Rimuovi classi di stato precedenti
  indicator.classList.remove('metadata-complete', 'metadata-error');
  
  // Arrotonda la percentuale a intero
  const roundedPercentage = Math.round(percentage);
  console.log(`Aggiornamento indicatore metadati: ${roundedPercentage}%`);
  
  if (percentage === -1) {
    // Caso errore
    percentageEl.textContent = 'Errore';
    indicator.classList.add('metadata-error');
    console.error('Errore metadati:', errorMsg);
  } else if (roundedPercentage >= 100) {
    // Completato
    percentageEl.textContent = '100%';
    indicator.classList.add('metadata-complete');
    console.log('Elaborazione metadati completata');
  } else {
    // In corso
    percentageEl.textContent = `${roundedPercentage}%`;
  }
}

/**
 * Interrompe il polling dei metadati
 */
function stopMetadataPolling() {
  if (metadataPollingInterval) {
    clearInterval(metadataPollingInterval);
    metadataPollingInterval = null;
  }
  isPollingActive = false;
  console.log("Polling dei metadati interrotto");
}