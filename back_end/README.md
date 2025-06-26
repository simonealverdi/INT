# BACK_END
AI Interview System
Panoramica del Progetto:
Il sistema AI Interview è una piattaforma completa per la conduzione di interviste automatizzate tramite intelligenza artificiale, composta da un frontend web moderno e un backend avanzato con architettura a microservizi.

Frontend (FRONT_END) ##------------
Tecnologie: JavaScript vaniglia, HTML5, CSS3, Vite come build tool
Caratteristiche principali:
Interfaccia utente intuitiva per caricare e gestire domande d'intervista
Sistema di autenticazione con login utente (username:admin password:admin)
Registrazione audio con rilevamento automatico del silenzio (VAD)
Riproduzione delle risposte vocali generate dall'AI
Visualizzazione animata dell'interazione vocale
Supporto per caricamento di file con domande (.docx, .csv, .xlsx, .json)
Backend (BACK_END) #------------
Tecnologie: Python, FastAPI, MongoDB, OpenAI API
Architettura:
Struttura modulare con API RESTful
Microservizi dedicati (NLP, TTS, trascrizione)
Sistema di persistenza dati con MongoDB
Funzionalità principali:
Autenticazione e gestione delle sessioni
Elaborazione del linguaggio naturale italiano
Sintesi vocale (TTS) con OpenAI o AWS Polly
Trascrizione audio con Whisper
Gestione delle domande e risposte d'intervista
Elaborazione dei file caricati dagli utenti
Punti Critici del Sistema
Integrazione e Comunicazione
Gestione delle richieste CORS tra frontend e backend
Corretta sincronizzazione tra registrazione audio e risposte AI
Latenza nella trasmissione e processamento dell'audio
Elaborazione del Linguaggio Naturale
Accuratezza dell'analisi NLP in lingua italiana
Ottimizzazione delle performance del microservizio NLP
Gestione di diversi accenti e dialetti regionali
Gestione Audio e Voce
Rilevamento affidabile dell'attività vocale (VAD)
Qualità e naturalezza della sintesi vocale
Ottimizzazione della dimensione e formato dei file audio
Scalabilità e Prestazioni
Gestione efficiente delle connessioni simultanee
Caching dei risultati TTS per ridurre latenza e costi
Ottimizzazione delle chiamate alle API esterne (OpenAI)
Sicurezza e Privacy
Protezione dei dati sensibili degli utenti
Gestione sicura delle chiavi API
Implementazione corretta dell'autenticazione
Dipendenze Esterne
Affidabilità delle API OpenAI e AWS
Gestione delle limitazioni di quota e rate limiting
Strategie di fallback in caso di indisponibilità dei servizi esterni
Implementazione Tecnica
Il sistema è progettato con un'architettura moderna e modulare che separa chiaramente le responsabilità tra frontend e backend. L'interazione avviene principalmente attraverso API RESTful, con il frontend che gestisce l'interfaccia utente e la registrazione audio, mentre il backend si occupa dell'elaborazione dei dati, dell'intelligenza artificiale e della generazione delle risposte.

La gestione dello stato dell'intervista viene mantenuta sul server, permettendo sessioni di intervista coerenti e persistenti anche in caso di disconnessione temporanea del client.

###---
Ti sottolineo che il primo passaggio per avviare il programma è avviare un sub-server uvicorn fastapi alla port 9000, installare i pacchetti elencati nel file di testo nlp_requirements.txt e avviare il tutto tramite prompt come segue 
python -m uvicorn app.run_nlp:app --port 9000.

In seguito, si può procedere ad avviare il main e il front_end: python Main.main:app  &&   npm run dev

Ovviamente bisogna installare anche i pacchettti elencati in requirements.txt
