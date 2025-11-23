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


# Modèles pour l'analyse de risque
class AnalyseRisqueInput(BaseModel):
    """Schéma pour l'analyse de risque de retard"""
    historique_taches_similaires: list[int] = Field(..., description="Liste des durées réelles passées (en jours)")
    complexite_actuelle: int = Field(..., ge=1, le=10, description="Complexité actuelle du chantier (entre 1 et 10)")


class AnalyseRisqueOutput(BaseModel):
    """Schéma de réponse pour l'analyse de risque"""
    risque_pourcentage: int = Field(..., ge=0, le=100, description="Pourcentage de risque de retard (0-100)")
    justification: str = Field(..., description="Justification du calcul du risque")


# Modèles pour la recommandation d'équipe
class RecommandationEquipeInput(BaseModel):
    """Schéma pour la recommandation d'équipe"""
    type_tache: str = Field(..., description="Type de tâche (ex: maçonnerie, plomberie, électricité)")
    complexite: int = Field(..., ge=1, le=10, description="Complexité de la tâche (entre 1 et 10)")


class RecommandationEquipeOutput(BaseModel):
    """Schéma de réponse pour la recommandation d'équipe"""
    equipe_recommandee: str = Field(..., description="Nom de l'équipe recommandée")
    performance_attendue: float = Field(..., ge=0, le=10, description="Performance attendue (0-10)")


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
    
    # Le chemin doit être relatif au répertoire où se trouve api.py
    model_path = Path(__file__).parent / "predictor.pt"
    
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
            "predict": "/predict (POST)",
            "analyse_risque": "/api/analyse_risque (POST)",
            "recommander_equipe": "/api/recommander_equipe (POST)"
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


@app.post("/api/analyse_risque", response_model=AnalyseRisqueOutput)
async def analyser_risque_retard(input_data: AnalyseRisqueInput):
    """
    Route POST pour analyser le risque de retard d'un chantier
    
    Args:
        input_data: Données d'entrée contenant historique_taches_similaires et complexite_actuelle
    
    Returns:
        AnalyseRisqueOutput: Pourcentage de risque et justification
    """
    try:
        historique = input_data.historique_taches_similaires
        complexite = input_data.complexite_actuelle
        
        if not historique or len(historique) == 0:
            # Si pas d'historique, risque basé uniquement sur la complexité
            risque = min(100, complexite * 10)
            justification = f"Pas d'historique disponible. Risque basé uniquement sur la complexité ({complexite}/10)."
            return AnalyseRisqueOutput(
                risque_pourcentage=int(risque),
                justification=justification
            )
        
        # Calculer la moyenne et l'écart-type de l'historique
        moyenne = np.mean(historique)
        ecart_type = np.std(historique) if len(historique) > 1 else 0
        
        # Calculer le nombre de dépassements (tâches qui ont pris plus de temps que prévu)
        # On suppose que la durée prévue est la moyenne
        depassements = sum(1 for duree in historique if duree > moyenne * 1.1)
        taux_depassement = depassements / len(historique) if len(historique) > 0 else 0
        
        # Calculer le risque : combinaison de la complexité, du taux de dépassement et de la variabilité
        risque_complexite = complexite * 8  # 0-80% basé sur complexité
        risque_depassement = taux_depassement * 50  # 0-50% basé sur historique
        risque_variabilite = min(30, ecart_type / moyenne * 100) if moyenne > 0 else 0  # Variabilité
        
        # Risque total (pondéré)
        risque_total = min(100, (risque_complexite * 0.4 + risque_depassement * 0.4 + risque_variabilite * 0.2))
        
        # Générer une justification
        justification_parts = []
        justification_parts.append(f"Complexité actuelle: {complexite}/10")
        justification_parts.append(f"Historique: {len(historique)} tâche(s) similaires analysée(s)")
        justification_parts.append(f"Taux de dépassement historique: {taux_depassement*100:.1f}%")
        if ecart_type > 0:
            justification_parts.append(f"Variabilité: {ecart_type/moyenne*100:.1f}%")
        justification_parts.append("Basé sur l'historique des tâches BTP.")
        
        justification = " ".join(justification_parts)
        
        return AnalyseRisqueOutput(
            risque_pourcentage=int(risque_total),
            justification=justification
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse de risque: {str(e)}"
        )


@app.post("/api/recommander_equipe", response_model=RecommandationEquipeOutput)
async def recommander_equipe(input_data: RecommandationEquipeInput):
    """
    Route POST pour recommander une équipe selon le type de tâche et la complexité
    
    Args:
        input_data: Données d'entrée contenant type_tache et complexite
    
    Returns:
        RecommandationEquipeOutput: Équipe recommandée et performance attendue
    """
    try:
        type_tache = input_data.type_tache.lower()
        complexite = input_data.complexite
        
        # Mapping des types de tâches vers les équipes recommandées
        equipes_par_type = {
            "maçonnerie": ["Équipe Alpha", "Équipe Beta"],
            "plomberie": ["Équipe Gamma", "Équipe Delta"],
            "électricité": ["Équipe Epsilon", "Équipe Zeta"],
            "charpente": ["Équipe Alpha", "Équipe Beta"],
            "couverture": ["Équipe Beta", "Équipe Gamma"],
            "isolation": ["Équipe Delta", "Équipe Epsilon"],
            "peinture": ["Équipe Zeta", "Équipe Alpha"],
            "carrelage": ["Équipe Gamma", "Équipe Delta"],
        }
        
        # Trouver l'équipe recommandée selon le type
        equipe_recommandee = "Équipe Standard"
        for keyword, equipes in equipes_par_type.items():
            if keyword in type_tache:
                # Si complexité élevée, prendre la première équipe (meilleure)
                # Si complexité faible, prendre la deuxième équipe
                equipe_recommandee = equipes[0] if complexite >= 7 else equipes[1] if len(equipes) > 1 else equipes[0]
                break
        
        # Si aucune correspondance, choisir selon la complexité
        if equipe_recommandee == "Équipe Standard":
            if complexite >= 8:
                equipe_recommandee = "Équipe Alpha"
            elif complexite >= 6:
                equipe_recommandee = "Équipe Gamma"
            elif complexite >= 4:
                equipe_recommandee = "Équipe Delta"
            else:
                equipe_recommandee = "Équipe Zeta"
        
        # Calculer la performance attendue (0-10)
        # Base: 7.0, ajustée selon la complexité et le type de tâche
        performance_base = 7.0
        
        # Bonus si l'équipe correspond bien au type de tâche
        bonus_correspondance = 1.5 if any(keyword in type_tache for keyword in equipes_par_type.keys()) else 0
        
        # Ajustement selon la complexité (tâches moyennes = meilleure performance)
        ajustement_complexite = 0.5 if 4 <= complexite <= 7 else -0.3
        
        performance_attendue = min(10.0, max(5.0, performance_base + bonus_correspondance + ajustement_complexite))
        
        return RecommandationEquipeOutput(
            equipe_recommandee=equipe_recommandee,
            performance_attendue=round(performance_attendue, 1)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la recommandation d'équipe: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

