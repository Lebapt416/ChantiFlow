"""
API FastAPI pour servir le modèle de prédiction de durée de chantier
Charge le modèle PyTorch et expose une route POST /predict
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

# Import du modèle (même structure que train_model.py)
class ChantierPredictor(nn.Module):
    """Réseau de neurones simple pour prédire la durée d'un chantier"""
    
    def __init__(self):
        super(ChantierPredictor, self).__init__()
        self.fc = nn.Linear(2, 1)
    
    def forward(self, x):
        return self.fc(x)


# Modèle Pydantic pour la validation des entrées
class ChantierInput(BaseModel):
    """Schéma de validation pour les données d'entrée"""
    nombre_taches: int = Field(..., ge=1, le=100, description="Nombre de tâches du chantier (entre 1 et 100)")
    complexite: float = Field(..., ge=1.0, le=10.0, description="Niveau de complexité du chantier (entre 1.0 et 10.0)")


# Modèle Pydantic pour la réponse
class ChantierPrediction(BaseModel):
    """Schéma de réponse pour la prédiction"""
    duree_estimee: float = Field(..., description="Durée estimée du chantier en jours")


# Initialiser FastAPI
app = FastAPI(
    title="ChantiFlow Prediction API",
    description="API pour prédire la durée d'un chantier basée sur le nombre de tâches et la complexité",
    version="1.0.0"
)

# Configurer CORS pour permettre les requêtes depuis le front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, remplacer par les origines spécifiques
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable globale pour stocker le modèle
model = None
mean_values = None
std_values = None


@app.on_event("startup")
async def load_model():
    """
    Charge le modèle PyTorch au démarrage de l'API
    """
    global model, mean_values, std_values
    
    model_path = Path("ml/predictor.pt")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Le modèle {model_path} n'existe pas. "
            "Veuillez d'abord exécuter train_model.py pour entraîner le modèle."
        )
    
    try:
        # Créer le modèle
        model = ChantierPredictor()
        
        # Charger les poids sauvegardés
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Mettre le modèle en mode évaluation
        model.eval()
        
        # Valeurs de normalisation (doivent correspondre à celles utilisées lors de l'entraînement)
        # Ces valeurs devraient être sauvegardées avec le modèle en production
        mean_values = torch.tensor([25.0, 5.5])  # Moyennes approximatives
        std_values = torch.tensor([12.0, 2.5])   # Écart-types approximatifs
        
        print(f"✅ Modèle chargé depuis {model_path}")
        print(f"📊 Modèle prêt à faire des prédictions")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        raise


@app.get("/")
async def root():
    """Route de base pour vérifier que l'API fonctionne"""
    return {
        "message": "ChantiFlow Prediction API",
        "status": "running",
        "endpoints": {
            "predict": "/predict (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """Route de santé pour vérifier l'état de l'API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=ChantierPrediction)
async def predict_chantier_duree(input_data: ChantierInput):
    """
    Route POST pour prédire la durée d'un chantier
    
    Args:
        input_data: Données d'entrée contenant nombre_taches et complexite
    
    Returns:
        ChantierPrediction: Durée estimée du chantier en jours
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas chargé. Veuillez réessayer plus tard."
        )
    
    try:
        # Convertir les entrées en tenseur
        input_tensor = torch.FloatTensor([[input_data.nombre_taches, input_data.complexite]])
        
        # Normaliser les entrées (même normalisation que lors de l'entraînement)
        if mean_values is not None and std_values is not None:
            input_tensor = (input_tensor - mean_values) / (std_values + 1e-8)
        
        # Faire la prédiction
        with torch.no_grad():
            prediction = model(input_tensor)
            duree_estimee = prediction.item()
        
        # S'assurer que la durée est positive
        duree_estimee = max(1.0, duree_estimee)
        
        return ChantierPrediction(duree_estimee=round(duree_estimee, 2))
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

