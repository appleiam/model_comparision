@echo off
REM Starts the FastAPI backend (port 8000) and the Vite frontend (port 5173) together.
setlocal
cd /d "%~dp0"

if not exist backend\.venv\Scripts\python.exe (
  echo [setup] creating backend\.venv ...
  python -m venv backend\.venv || goto :error
  backend\.venv\Scripts\python -m pip install --upgrade pip
  backend\.venv\Scripts\python -m pip install -r backend\requirements.txt || goto :error
)
if not exist frontend\node_modules (
  echo [setup] npm install ...
  pushd frontend && call npm install && popd
)

start "APCS backend" cmd /k "cd /d %~dp0backend && .venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
start "APCS frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
echo.
echo Backend:  http://127.0.0.1:8000/system
echo Frontend: http://localhost:5173
timeout /t 4 >nul
start "" http://localhost:5173
goto :eof

:error
echo Setup failed. See README.md.
exit /b 1
