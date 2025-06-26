// ------------------------------------------------------------
// Config
// ------------------------------------------------------------
const BACKEND = "http://127.0.0.1:8000";

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const questionFile = document.getElementById('questionFile');
    const fileNameDisplay = document.getElementById('fileName');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    const startInterview = document.getElementById('startInterview');
    
    // Check if session already exists
    checkSession();
    
    // Display file name when selected
    questionFile.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            fileNameDisplay.textContent = this.files[0].name;
            uploadBtn.disabled = false;
        } else {
            fileNameDisplay.textContent = 'Nessun file selezionato';
            uploadBtn.disabled = true;
        }
    });
    
    // Upload questions when button clicked
    uploadBtn.addEventListener('click', function() {
        uploadQuestions();
    });
    
    /**
     * Upload questions to the backend
     */
    async function uploadQuestions() {
        if (!questionFile.files || !questionFile.files[0]) {
            showStatus('Seleziona un file prima di caricare.', 'error');
            return;
        }
        
        // Show loading status
        showStatus('Caricamento in corso...', '');
        uploadBtn.disabled = true;
        
        const formData = new FormData();
        formData.append('file', questionFile.files[0]);
        
        try {
            const response = await fetch(`${BACKEND}/load_questions`, {
                method: 'POST',
                body: formData,
                credentials: 'include' // Important for cookies
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                // Enable interview button
                startInterview.classList.remove('disabled');
                showStatus(`${result.count} domande caricate con successo!`, 'success');
            } else {
                showStatus(`Errore: ${result.message}`, 'error');
                uploadBtn.disabled = false;
            }
        } catch (error) {
            showStatus('Errore di connessione al server. Riprova più tardi.', 'error');
            console.error('Upload error:', error);
            uploadBtn.disabled = false;
        }
    }
    
    /**
     * Check if a valid session exists
     */
    async function checkSession() {
        try {
            const response = await fetch(`${BACKEND}/check_session`, {
                method: 'GET',
                credentials: 'include' // Important for cookies
            });
            
            const result = await response.json();
            
            if (result.valid && result.questions_loaded) {
                // Questions already loaded, enable interview button
                startInterview.classList.remove('disabled');
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
});
