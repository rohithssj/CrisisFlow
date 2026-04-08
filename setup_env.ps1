# ⚡ CrisisFlow - Local Environment Setup Script (Windows)

# 1. Check if Python 3.11 is installed
Write-Host "🔍 checking for Python 3.11..." -ForegroundColor Cyan
$py311 = py --list-paths | Select-String "-V:3.11"
if (-not $py311) {
    Write-Host "❌ Python 3.11 not found! Please download it from python.org" -ForegroundColor Red
    exit
}

# 2. Create Virtual Environment
Write-Host "🛠️ Creating a new virtual environment using Python 3.11..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "⚠️ Existing 'venv' found. Deleting for a clean reinstall..." -ForegroundColor Yellow
    Remove-Item -Path "venv" -Recurse -Force
}
py -3.11 -m venv venv

# 3. Upgrade Pip and Install Requirements
Write-Host "📦 Installing dependencies from requirements.txt..." -ForegroundColor Cyan
& ".\venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\venv\Scripts\pip.exe" install -r requirements.txt --no-cache-dir

# 4. Verify Installation
Write-Host "✅ Verifying pandas installation..." -ForegroundColor Green
& ".\venv\Scripts\python.exe" -c "import pandas; print('Pandas Version: ' + pandas.__version__)"

Write-Host "`n🚀 Setup complete! To start the dashboard, run:" -ForegroundColor Green
Write-Host ".\venv\Scripts\activate" -ForegroundColor Yellow
Write-Host "streamlit run app.py" -ForegroundColor Yellow
