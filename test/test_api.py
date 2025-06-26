import requests
import json

# Semplice script per testare gli endpoint API

BASE_URL = 'http://127.0.0.1:8000'

def test_endpoint(url, method='GET', data=None):
    print(f"\nTestando {method} {url}")
    try:
        if method == 'GET':
            response = requests.get(url)
        elif method == 'POST':
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Metodo {method} non supportato")
            
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            try:
                print("Risposta JSON:")
                print(json.dumps(response.json(), indent=2))
            except:
                print("Risposta non JSON:")
                print(response.text[:200] + '...' if len(response.text) > 200 else response.text)
        else:
            print(f"Errore: {response.text}")
    except Exception as e:
        print(f"Errore nel test: {e}")

# Test endpoints
print("\n=== TEST DELLE API PRINCIPALI ===")
test_endpoint(f"{BASE_URL}/api/tts/available_voices")

print("\n=== TEST DELL'AUTENTICAZIONE ===")
login_data = {"username": "admin", "password": "admin"}  # Credenziali semplificate per sviluppo
test_endpoint(f"{BASE_URL}/api/token", method='POST', data=login_data)

print("\n=== TEST DEL FIRST PROMPT ===")
test_endpoint(f"{BASE_URL}/api/first_prompt?user_id=utente_demo")

print("\n=== TEST DELLA SESSIONE ===")
test_endpoint(f"{BASE_URL}/api/check_session")
