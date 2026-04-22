# E-SAFE Run Script for PowerShell
# 에스컬레이터 안전 AI 시스템 실행 스크립트

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  E-SAFE: Escalator Safety AI System" -ForegroundColor Cyan
Write-Host "  인천공항 에스컬레이터 안전 모니터링" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[✓] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[✗] Python not found. Please install Python 3.10+" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if pip exists
try {
    python -m pip --version | Out-Null
    Write-Host "[✓] pip found" -ForegroundColor Green
} catch {
    Write-Host "[✗] pip not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[2/4] Checking dependencies..." -ForegroundColor Yellow

# Check critical dependencies
$dependencies = @("streamlit", "cv2", "numpy", "ultralytics", "supervision")
$missingDeps = @()

foreach ($dep in $dependencies) {
    try {
        python -c "import $dep" 2>$null
        Write-Host "[✓] $dep installed" -ForegroundColor Green
    } catch {
        Write-Host "[✗] $dep missing" -ForegroundColor Yellow
        $missingDeps += $dep
    }
}

if ($missingDeps.Count -gt 0) {
    Write-Host ""
    Write-Host "[WARNING] Installing missing dependencies..." -ForegroundColor Yellow
    
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
    $reqFile = Join-Path $scriptPath "requirements.txt"
    
    if (Test-Path $reqFile) {
        python -m pip install -r $reqFile --quiet
        Write-Host "[✓] Dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "[✗] requirements.txt not found" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "[3/4] Checking GPU availability..." -ForegroundColor Yellow
try {
    $gpuStatus = python -c "import torch; print('GPU available' if torch.cuda.is_available() else 'GPU not available')" 2>$null
    Write-Host "[Note] $gpuStatus" -ForegroundColor Gray
} catch {
    Write-Host "[Note] GPU check skipped (PyTorch not required)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[4/4] Starting E-SAFE Streamlit Dashboard..." -ForegroundColor Cyan
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Dashboard URL: http://localhost:8501" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
streamlit run app.py
Pop-Location

Write-Host ""
Write-Host "Streamlit dashboard closed." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
