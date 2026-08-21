@echo off
echo ========================================
echo  Voice-Enabled RAG - Setup & Run
echo  HH Goa 2026 - Task 2
echo  LLM: NVIDIA NIM (Free)
echo  STT: Whisper (Local, Free)
echo ========================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    exit /b 1
)

REM Check .env
if not exist .env (
    echo.
    echo WARNING: .env file not found!
    echo Copy .env.example to .env and add your NVIDIA API key
    echo Get free key at: https://build.nvidia.com
    echo.
    pause
)

REM Install dependencies
echo [1/5] Installing dependencies...
pip install -r requirements.txt

REM Download NLTK data
echo [2/5] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

REM Download Whisper model
echo [3/5] Downloading Whisper model (base)...
python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('Whisper model ready')"

REM Ingest data
echo [4/5] Ingesting dataset...
python -m data.ingest

REM Start server
echo [5/5] Starting API server...
echo.
echo API:     http://localhost:8000
echo Docs:    http://localhost:8000/docs
echo UI:      streamlit run frontend/app.py
echo.
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
