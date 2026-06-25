@echo off
cd /d "%~dp0"
echo Starting CreditPulse ML Dashboard...
start "CreditPulse Server" cmd /k ""C:\Users\hp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" server.py"
timeout /t 4 /nobreak >nul
start "" "http://localhost:8000"
exit
