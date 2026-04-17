@echo off
:: ──────────────────────────────────────────────────────────────────────────────
::  DocumentLens NLP Summarizer – Windows Setup & Run Script
:: ──────────────────────────────────────────────────────────────────────────────

echo.
echo  DocumentLens - NLP Document Summarizer
echo ==========================================

where python >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo  ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)

echo  Found Python:
python --version

echo.
echo  Creating virtual environment...
python -m venv venv

echo  Activating virtual environment...
call venv\Scripts\activate.bat

echo  Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo.
echo  Setup complete! Starting server...
echo  ================================================
echo    Open browser at: http://127.0.0.1:5000
echo    Press Ctrl+C to stop
echo  ================================================
echo.
python app.py
pause
