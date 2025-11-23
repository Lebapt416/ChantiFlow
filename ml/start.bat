@echo off
REM Script de démarrage de l'API FastAPI pour ChantiFlow (Windows)

echo 🚀 Démarrage de l'API FastAPI ChantiFlow...

REM Vérifier que nous sommes dans le bon dossier
if not exist "api.py" (
    echo ❌ Erreur: Ce script doit être exécuté depuis le dossier 'ml'
    echo    Exécutez: cd ml && start.bat
    exit /b 1
)

REM Vérifier que Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Erreur: Python n'est pas installé
    exit /b 1
)

REM Vérifier que les dépendances sont installées
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installation des dépendances Python...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Erreur lors de l'installation des dépendances
        exit /b 1
    )
)

REM Vérifier que le modèle existe
if not exist "predictor.pt" (
    echo 🤖 Entraînement du modèle...
    python train_model.py
    if errorlevel 1 (
        echo ❌ Erreur lors de l'entraînement du modèle
        exit /b 1
    )
)

REM Démarrer l'API
echo ✅ Démarrage de l'API sur http://localhost:8000
echo    Appuyez sur CTRL+C pour arrêter
echo.

python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000

