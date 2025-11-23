#!/bin/bash

# Script de démarrage de l'API FastAPI pour ChantiFlow

echo "🚀 Démarrage de l'API FastAPI ChantiFlow..."

# Obtenir le répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Vérifier que nous sommes dans le bon dossier
if [ ! -f "api.py" ]; then
    echo "❌ Erreur: Le fichier api.py est introuvable"
    echo "   Répertoire actuel: $(pwd)"
    echo "   Exécutez ce script depuis le dossier 'ml'"
    exit 1
fi

echo "📁 Répertoire de travail: $(pwd)"

# Vérifier que Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Erreur: Python 3 n'est pas installé"
    exit 1
fi

# Vérifier que les dépendances sont installées
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📦 Installation des dépendances Python..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Erreur lors de l'installation des dépendances"
        exit 1
    fi
fi

# Vérifier que le modèle existe
if [ ! -f "predictor.pt" ]; then
    echo "🤖 Entraînement du modèle..."
    python3 train_model.py
    if [ $? -ne 0 ]; then
        echo "❌ Erreur lors de l'entraînement du modèle"
        exit 1
    fi
fi

# Démarrer l'API
echo "✅ Démarrage de l'API sur http://localhost:8000"
echo "   Répertoire: $(pwd)"
echo "   Appuyez sur CTRL+C pour arrêter"
echo ""

python3 -m uvicorn api:app --reload --host 0.0.0.0 --port 8000

