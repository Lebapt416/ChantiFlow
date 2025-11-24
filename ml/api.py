from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import csv
import os
from typing import List

# --- CONFIGURATION ---
# Utilisation de /data pour le volume persistant si dispo, sinon dossier local
BASE_DIR = Path("/data") if os.path.exists("/data") else Path(__file__).parent
MODEL_PATH = BASE_DIR / "predictor.pt"
DATA_PATH = BASE_DIR / "real_data.csv"
RETRAIN_EVERY_N_SAMPLES = 5


class ChantierPredictor(nn.Module):
    def __init__(self):
        super(ChantierPredictor, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):  # type: ignore[override]
        return self.fc(x)


class ChantierInput(BaseModel):
    nombre_taches: int
    complexite: float


class ChantierPrediction(BaseModel):
    duree_estimee: float


class FeedbackInput(BaseModel):
    nombre_taches: int
    complexite: float
    duree_reelle: float


class SiteData(BaseModel):
    name: str
    tasks_total: int
    tasks_done: int
    complexity: float
    days_elapsed: int


class GlobalSummaryInput(BaseModel):
    sites: List[SiteData]


class SiteSummaryInput(BaseModel):
    site_name: str
    tasks_total: int
    tasks_pending: int
    complexity: float
    days_elapsed: int
    planned_duration: int


class SummaryResponse(BaseModel):
    summary: str
    status: str


app = FastAPI(title="ChantiFlow Self-Learning AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
new_samples_counter = 0


def get_norm_params():
    return torch.tensor([25.0, 5.5]), torch.tensor([12.0, 2.5])


def train_on_new_data():
    global model
    print("🧠 Ré-entraînement en cours...")
    if not DATA_PATH.exists():
        return

    data_x, data_y = [], []
    try:
        with open(DATA_PATH, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    data_x.append([float(row[0]), float(row[1])])
                    data_y.append([float(row[2])])
    except Exception as e:
        print(f"Erreur lecture CSV: {e}")
        return

    if len(data_x) < 5:
        return

    X = torch.FloatTensor(data_x)
    y = torch.FloatTensor(data_y)

    mean_v, std_v = get_norm_params()
    X_norm = (X - mean_v) / (std_v + 1e-8)

    local_model = ChantierPredictor()
    if MODEL_PATH.exists():
        try:
            local_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu")["model_state_dict"])
        except Exception:
            pass

    optimizer = optim.SGD(local_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    local_model.train()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(local_model(X_norm), y)
        loss.backward()
        optimizer.step()

    torch.save({"model_state_dict": local_model.state_dict()}, MODEL_PATH)

    model = local_model
    model.eval()
    print(f"✅ Modèle mis à jour. Loss: {loss.item():.4f}")


@app.on_event("startup")
async def load_model():
    global model
    model = ChantierPredictor()

    if MODEL_PATH.exists():
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu")["model_state_dict"])
            print("✅ Modèle chargé.")
        except Exception:
            print("⚠️ Modèle corrompu ou incompatible, démarrage à zéro.")

    model.eval()


@app.get("/health")
async def health_check():
    """Route de santé pour vérifier l'état de l'API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "learning_enabled": True,
    }


@app.post("/summary/global", response_model=SummaryResponse)
async def generate_global_summary(data: GlobalSummaryInput):
    """Génère un résumé de l'état de santé de tous les chantiers"""
    if not data.sites:
        return SummaryResponse(summary="Aucun chantier actif pour le moment.", status="good")

    total_sites = len(data.sites)
    sites_at_risk = 0
    sites_critical = 0
    mean_v, std_v = get_norm_params()

    for site in data.sites:
        if model is not None:
            with torch.no_grad():
                inp = torch.FloatTensor([[site.tasks_total, site.complexity]])
                inp_norm = (inp - mean_v) / (std_v + 1e-8)
                predicted_duration = model(inp_norm).item()
        else:
            predicted_duration = site.tasks_total * 0.5

        progress_ratio = site.tasks_done / max(1, site.tasks_total)
        time_ratio = site.days_elapsed / max(1, predicted_duration)

        if time_ratio > 0.8 and progress_ratio < 0.5:
            sites_critical += 1
        elif time_ratio > 0.6 and progress_ratio < 0.4:
            sites_at_risk += 1

    if sites_critical > 0:
        text = f"⚠️ Attention requise : {sites_critical} chantier(s) en situation critique par rapport aux prédictions IA."
        status = "critical"
    elif sites_at_risk > 0:
        text = f"🟠 Vigilance : {sites_at_risk} chantier(s) montrent des signes de ralentissement d'après l'analyse."
        status = "warning"
    else:
        text = f"✨ Tout va bien. Les {total_sites} chantiers avancent conformément aux estimations de l'IA."
        status = "good"

    return SummaryResponse(summary=text, status=status)


@app.post("/summary/site", response_model=SummaryResponse)
async def generate_site_summary(data: SiteSummaryInput):
    """Génère un résumé pour un chantier spécifique"""
    if model is not None:
        mean_v, std_v = get_norm_params()
        with torch.no_grad():
            inp = torch.FloatTensor([[data.tasks_total, data.complexity]])
            inp_norm = (inp - mean_v) / (std_v + 1e-8)
            predicted_total_days = max(1, model(inp_norm).item())
    else:
        predicted_total_days = data.tasks_total * 0.8

    retard_estime = predicted_total_days - data.planned_duration
    progression = 1 - (data.tasks_pending / max(1, data.tasks_total))

    if retard_estime > 5:
        status = "critical"
        text = f"⚠️ Risque élevé : L'IA prévoit {int(predicted_total_days)} jours de travail (vs {data.planned_duration} prévus). Un retard de {int(retard_estime)} jours est probable."
    elif retard_estime > 2:
        status = "warning"
        text = f"🟠 Attention : Le rythme actuel suggère un léger dépassement ({int(retard_estime)} jours) par rapport au planning."
    elif progression > 0.9:
        status = "good"
        text = "✅ Chantier en phase de finition. Les objectifs sont atteints."
    else:
        status = "good"
        text = f"✨ Le chantier avance normalement. L'estimation IA ({int(predicted_total_days)}j) est alignée avec votre planning."

    return SummaryResponse(summary=text, status=status)


@app.post("/predict", response_model=ChantierPrediction)
async def predict(input_data: ChantierInput):
    if model is None:
        raise HTTPException(503, "Modèle non chargé")

    mean_v, std_v = get_norm_params()
    inp = torch.FloatTensor([[input_data.nombre_taches, input_data.complexite]])
    inp_norm = (inp - mean_v) / (std_v + 1e-8)

    with torch.no_grad():
        pred = model(inp_norm).item()

    return ChantierPrediction(duree_estimee=max(1.0, round(pred, 2)))


@app.post("/feedback")
async def feedback(data: FeedbackInput, bg_tasks: BackgroundTasks):
    global new_samples_counter

    file_exists = DATA_PATH.exists()
    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["taches", "complexite", "duree"])
        writer.writerow([data.nombre_taches, data.complexite, data.duree_reelle])

    new_samples_counter += 1
    if new_samples_counter >= RETRAIN_EVERY_N_SAMPLES:
        bg_tasks.add_task(train_on_new_data)
        new_samples_counter = 0
        return {"status": "training_scheduled"}

    return {"status": "saved"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
