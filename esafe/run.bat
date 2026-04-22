@echo off
REM E-SAFE Run Script for Windows
REM 에스컬레이터 안전 AI 시스템 실행 스크립트

echo.
echo ========================================
echo  E-SAFE: Escalator Safety AI System
echo  인천공항 에스컬레이터 안전 모니터링
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version

REM Check if pip exists
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip not found
    pause
    exit /b 1
)

echo.
echo [2/4] Checking dependencies...
python -c "import streamlit; import cv2; import numpy; import ultralytics" 2>nul
if errorlevel 1 (
    echo [WARNING] Some dependencies missing. Installing...
    cd /d "%~dp0"
    if exist requirements.txt (
        python -m pip install -r requirements.txt --quiet
    ) else (
        echo [ERROR] requirements.txt not found
        pause
        exit /b 1
    )
)

echo [✓] All dependencies OK

echo.
echo [3/4] Checking GPU availability...
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')" 2>nul || echo [Note] GPU check skipped

echo.
echo [4/4] Starting E-SAFE Streamlit Dashboard...
echo.
echo ========================================
echo  Dashboard URL: http://localhost:8501
echo  Press Ctrl+C to stop
echo ========================================
echo.

cd /d "%~dp0"
streamlit run app.py

pause
