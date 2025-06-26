print("ciao")

from memoria import salva_memoria


testo = "I enjoy hiking in the mountains and spending time with my family."
keywords = ["hiking", "mountains", "family"]

salva_memoria(user_id="test_user", testo=testo, keyword=keywords)

print("✅ Documento inserito!")