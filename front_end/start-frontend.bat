@echo off
echo Avvio server di sviluppo per il frontend...
cd "%~dp0"

echo Utilizzando server HTTP semplice sulla porta 5500
python -m http.server 5500

echo Server terminato.
