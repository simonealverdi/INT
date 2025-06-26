// voice_chat.js – front‑end vanilla al 100 %
// ------------------------------------------------------------
// Config
// ------------------------------------------------------------
const BACKEND  = "http://127.0.0.1:8000";
const USER_ID  = "utente_demo";           // di solito viene dallo login

// Ridotto a 1.5s per interazione più rapida
const SILENCE_MS = 5000;                 // 2s di silenzio ⇒ fine turno
const startBtn = document.getElementById("start");
const illustration = document.getElementById("illustration");

// Stato globale minimale
let mediaRecorder, audioCtx, analyser;
let chunks = [];
let isListening = false;
let isAISpeaking = false;
let sessionUserId = null;  // Memorizza l'ID utente per la sessione corrente

// Nuove variabili per gestione pausa / ripresa
let conversationStarted = false;  // true dopo il primissimo avvio
let paused = false;               // true se la conversazione è in pausa
let currentAudio = null;          // riferimento all'audio TTS in riproduzione

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
async function playRemoteAudio(url) {
  console.log("Richiesta audio:", url);
  
  try {
    // Se stiamo già riproducendo, interrompiamo
    if (currentAudio) {
      currentAudio.pause();
      currentAudio = null;
    }
    
    // Se siamo in pausa, aspettiamo una ripresa
    if (paused) {
      await waitForResume();
    }
    
    // Fetch dell'audio con timeout
    const audioResponse = await fetch(url);
    
    if (!audioResponse.ok) {
      throw new Error(`Errore HTTP: ${audioResponse.status}`);
    }
    
    const blob = await audioResponse.blob();
    
    if (blob.size < 100) {
      throw new Error("Audio file vuoto o troppo piccolo");
    }
    
    // Log dimensione e tipo
    console.log(`Audio ricevuto: ${blob.size} bytes, tipo: ${blob.type}`);
    
    // Crea elemento audio e riproduci
    const audio = new Audio(URL.createObjectURL(blob));
    currentAudio = audio;
    
    // Visualizza l'ondulazione mentre l'AI parla
    document.querySelector('#illustration').classList.add('speaking');
    
    audio.onerror = (e) => {
      console.error("Errore riproduzione:", e);
      document.querySelector('#illustration').classList.remove('speaking');
    };
    
    audio.onended = () => {
      console.log("Audio terminato");
      currentAudio = null;
      document.querySelector('#illustration').classList.remove('speaking');
    };
    
    await audio.play();
    return new Promise(resolve => {
      audio.onended = () => {
        currentAudio = null;
        document.querySelector('#illustration').classList.remove('speaking');
        resolve();
      };
    });
  } catch (err) {
    console.error("Errore riproduzione audio:", err);
    document.querySelector('#illustration').classList.remove('speaking');
    currentAudio = null;
    return Promise.resolve(); // Continuiamo l'esecuzione anche in caso di errore
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
    if (rms > 0.1) {
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

  try {
    // Mostra indicatore di caricamento se non è già visibile
    const thinkingEl = document.getElementById('thinking');
    if (thinkingEl && thinkingEl.classList.contains('hidden')) {
      thinkingEl.classList.remove('hidden');
    }
    
    const res = await fetch(`${BACKEND}/transcribe`, { method: "POST", body: form });
    const data = await res.json(); // { audio_url, type }

    console.log("➡️  nuova domanda (" + data.type + ")");
    console.log(BACKEND + data.audio_url)
    console.log(paused, currentAudio)
    // riproduci la nuova domanda e ricomincia il ciclo
    // Se data.audio_url inizia con /speak?text=... decodifica e ricodifica il testo
    let audioUrl = data.audio_url;
    
    // Se l'URL contiene già il dominio (http), non aggiungere BACKEND
    if (audioUrl.startsWith('http')) {
      console.log('URL assoluto rilevato, non aggiungo BACKEND');
      await playRemoteAudio(audioUrl);
    } else {
      // Se è un URL relativo, dobbiamo aggiungere il dominio
      if (audioUrl.startsWith('/speak?text=')) {
        const testo = decodeURIComponent(audioUrl.split('=')[1] || '');
        audioUrl = '/speak?text=' + encodeURIComponent(testo);
      }
      console.log('URL relativo rilevato, aggiungo BACKEND: ' + BACKEND + audioUrl);
      await playRemoteAudio(BACKEND + audioUrl);
    }
    await startRecording();

  } catch (err) {
    console.error("Upload error:", err);
    alert("Errore backend: vedi console.");
    resetUI();
  }
}

/* ------------------------------------------------------------
 * 4.  KICK‑OFF iniziale
 * -----------------------------------------------------------*/
async function getFirstPromptAudioURL() {
  // Usa l'endpoint first_prompt per ottenere la prima domanda strutturata
  try {
    const userId = ensureSessionUserId();  // Usa la funzione helper
    const response = await fetch(`${BACKEND}/api/first_prompt?user_id=${encodeURIComponent(userId)}`);
    if (!response.ok) throw new Error('Errore nel recupero del primo prompt');
    
    const data = await response.json();
    return data.audio_url;
  } catch (error) {
    console.error('Errore nel recupero del primo prompt:', error);
    throw error;
  }
}

// Avvia la conversazione dall'inizio
async function startConversation() {
  try {
    // UI feedback immediato
    startBtn.classList.add('recording');

    const firstAudioURL = await getFirstPromptAudioURL();
    await playRemoteAudio(firstAudioURL);  // pronuncia la prima domanda
    await startRecording();                // avvia registrazione utente
  } catch (err) {
    console.error("Error starting conversation:", err);
    alert("Errore nell'avvio della conversazione. Controlla che il server sia in esecuzione.");
    resetUI();
    conversationStarted = false;
  }
}

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
  if (!conversationStarted) {
    conversationStarted = true;
    await startConversation();
    return;
  }

  // Toggle pausa / ripresa
  if (!paused) {
    pauseConversation();
  } else {
    resumeConversation();
  }
}

// Eventi UI
document.addEventListener('DOMContentLoaded', async function() {
  // Verify authorization before initializing 
  const isAuthorized = await checkAuthorization();
  if (!isAuthorized) return;
  
  startBtn.addEventListener("click", handleMicButton);

  // Gestione pulsante impostazioni
  document.getElementById('settings')?.addEventListener('click', () => {
    alert('Settings dialog (placeholder)');
  });

  // Crea l'indicatore "Sto pensando..." se non esiste
  if (!document.getElementById('thinking')) {
    const thinkingEl = document.createElement('div');
    thinkingEl.id = 'thinking';
    thinkingEl.className = 'thinking hidden';
    thinkingEl.textContent = 'Sto pensando...';
    document.body.appendChild(thinkingEl);
  }
});

// Check session authorization first
async function checkAuthorization() {
  try {
    const response = await fetch(`${BACKEND}/api/check_session`, {
      method: 'GET',
      credentials: 'include' // Important for cookies
    });
    
    const result = await response.json();
    
    // If no valid session with questions, redirect to home
    if (!result.valid || !result.questions_loaded) {
      window.location.href = 'index.html';
      return false;
    }
    return true;
  } catch (error) {
    console.error('Session validation failed:', error);
    window.location.href = 'index.html';
    return false;
  }
}