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
from typing import List, Optional
import requests
from datetime import datetime, timedelta

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
    location: Optional[str] = None  # Code postal du chantier pour la météo


class SummaryResponse(BaseModel):
    summary: str
    status: str
    sites_mentioned: Optional[List[str]] = None  # Noms des chantiers mentionnés


class WeatherData(BaseModel):
    temperature: float
    precipitation: float
    weather_code: int
    date: str


class TaskWeatherInput(BaseModel):
    task_role: Optional[str] = None
    task_title: str
    planned_date: str
    location: Optional[str] = None  # Ville ou coordonnées


class WeatherOptimizationInput(BaseModel):
    tasks: List[TaskWeatherInput]
    location: Optional[str] = None
    start_date: str


class WeatherRecommendation(BaseModel):
    date: str
    temperature: float
    precipitation: float
    favorable: bool
    reason: str
    recommendation: str


class OptimizedPlanningResponse(BaseModel):
    recommendations: List[WeatherRecommendation]
    best_dates: List[str]
    warnings: List[str]


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


def geocode_location(location: str) -> Optional[tuple[float, float]]:
    """Géocode un code postal ou une ville en coordonnées lat/lon, avec préférence pour la France"""
    if not location or not location.strip():
        return None
    
    location_clean = location.strip()
    
    # Vérifier si c'est un code postal (5 chiffres)
    is_postal_code = location_clean.isdigit() and len(location_clean) == 5
    
    try:
        if is_postal_code:
            # Utiliser l'API Géo du gouvernement français pour les codes postaux
            response = requests.get(
                f"https://geo.api.gouv.fr/communes?codePostal={location_clean}&limit=1&fields=nom,code,centre",
                headers={"Accept": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0 and data[0].get("centre"):
                    # L'API retourne [longitude, latitude]
                    coords = data[0]["centre"]["coordinates"]
                    return (float(coords[1]), float(coords[0]))  # (lat, lon)
        else:
            # Pour les villes, utiliser Nominatim
            # Si la ville ne contient pas "France", l'ajouter pour améliorer la précision
            if "france" not in location_clean.lower():
                location_clean = f"{location_clean}, France"
            
            response = requests.get(
                f"https://nominatim.openstreetmap.org/search",
                params={
                    "q": location_clean,
                    "format": "json",
                    "limit": 1,
                    "countrycodes": "fr",  # Restreindre à la France
                    "addressdetails": 1
                },
                headers={"User-Agent": "ChantiFlowApp/1.0 (contact@chantiflow.com)"},
                timeout=5
            )
            if response.ok:
                data = response.json()
                if data and len(data) > 0:
                    return (float(data[0]["lat"]), float(data[0]["lon"]))
        
        return None
    except Exception as e:
        print(f"Erreur géocodage pour '{location}': {e}")
        return None


def get_weather_forecast(lat: float, lon: float, days: int = 7) -> List[WeatherData]:
    """Récupère les prévisions météo depuis OpenMeteo"""
    try:
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,precipitation_sum,weathercode",
                "timezone": "Europe/Paris",
                "forecast_days": days
            },
            timeout=5
        )
        if response.ok:
            data = response.json()
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            temps = daily.get("temperature_2m_max", [])
            precip = daily.get("precipitation_sum", [])
            codes = daily.get("weathercode", [])
            
            return [
                WeatherData(
                    temperature=temps[i],
                    precipitation=precip[i],
                    weather_code=codes[i],
                    date=dates[i]
                )
                for i in range(min(len(dates), days))
            ]
    except Exception as e:
        print(f"Erreur météo: {e}")
    return []


def check_weather_for_role(role: Optional[str], weather: WeatherData) -> dict:
    """Vérifie si les conditions météo sont favorables pour un métier"""
    # Règles simplifiées (version backend, règles complètes dans work-rules.ts)
    rules = {
        "maçon": {"avoid_rain": True, "min_temp": 5},
        "carreleur": {"avoid_rain": True, "min_temp": 5},
        "charpentier": {"avoid_rain": True, "min_temp": 0},
        "peintre": {"avoid_rain": True, "min_temp": 10, "max_temp": 30},
        "terrassier": {"avoid_rain": True, "min_temp": 0},
        "couvreur": {"avoid_rain": True, "min_temp": 5, "max_temp": 30},
    }
    
    if not role:
        return {"favorable": True, "reason": "Pas de contrainte météo spécifique"}
    
    role_lower = role.lower()
    rule = None
    for key, r in rules.items():
        if key in role_lower:
            rule = r
            break
    
    if not rule:
        return {"favorable": True, "reason": "Métier sans contrainte météo spécifique"}
    
    reasons = []
    if rule.get("avoid_rain") and weather.precipitation > 0.5:
        reasons.append(f"Pluie prévue ({weather.precipitation:.1f}mm)")
    if rule.get("min_temp") and weather.temperature < rule["min_temp"]:
        reasons.append(f"Température trop basse ({weather.temperature:.1f}°C)")
    if rule.get("max_temp") and weather.temperature > rule["max_temp"]:
        reasons.append(f"Température trop élevée ({weather.temperature:.1f}°C)")
    
    if reasons:
        return {
            "favorable": False,
            "reason": " | ".join(reasons),
            "recommendation": f"Reporter cette tâche à une date plus favorable"
        }
    
    return {
        "favorable": True,
        "reason": "Conditions météo favorables",
        "recommendation": "Date idéale pour cette tâche"
    }


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
        return SummaryResponse(summary="Aucun chantier actif pour le moment.", status="good", sites_mentioned=[])

    total_sites = len(data.sites)
    sites_at_risk = []
    sites_critical = []
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
            sites_critical.append(site.name)
        elif time_ratio > 0.6 and progress_ratio < 0.4:
            sites_at_risk.append(site.name)

    sites_mentioned = sites_critical + sites_at_risk
    
    if len(sites_critical) > 0:
        if len(sites_critical) == 1:
            text = f"⚠️ Attention requise : Le chantier « {sites_critical[0]} » est en situation critique par rapport aux prédictions IA."
        elif len(sites_critical) <= 3:
            sites_list = "« " + " », « ".join(sites_critical) + " »"
            text = f"⚠️ Attention requise : {len(sites_critical)} chantiers en situation critique : {sites_list}."
        else:
            sites_list = "« " + " », « ".join(sites_critical[:3]) + " »"
            text = f"⚠️ Attention requise : {len(sites_critical)} chantiers en situation critique, notamment : {sites_list}."
        status = "critical"
    elif len(sites_at_risk) > 0:
        if len(sites_at_risk) == 1:
            text = f"🟠 Vigilance : Le chantier « {sites_at_risk[0]} » montre des signes de ralentissement d'après l'analyse IA."
        elif len(sites_at_risk) <= 3:
            sites_list = "« " + " », « ".join(sites_at_risk) + " »"
            text = f"🟠 Vigilance : {len(sites_at_risk)} chantiers montrent des signes de ralentissement : {sites_list}."
        else:
            sites_list = "« " + " », « ".join(sites_at_risk[:3]) + " »"
            text = f"🟠 Vigilance : {len(sites_at_risk)} chantiers montrent des signes de ralentissement, notamment : {sites_list}."
        status = "warning"
    else:
        if total_sites == 1:
            text = "✨ Tout va bien. Le chantier avance conformément aux estimations de l'IA."
        else:
            text = f"✨ Tout va bien. Les {total_sites} chantiers avancent conformément aux estimations de l'IA."
        status = "good"

    return SummaryResponse(summary=text, status=status, sites_mentioned=sites_mentioned)


@app.post("/summary/site", response_model=SummaryResponse)
async def generate_site_summary(data: SiteSummaryInput):
    """Génère un résumé pour un chantier spécifique avec recommandations météo"""
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

    base_text = ""
    site_name_ref = f"« {data.site_name} »"
    
    if retard_estime > 5:
        status = "critical"
        base_text = f"⚠️ Risque élevé pour {site_name_ref} : L'IA prévoit {int(predicted_total_days)} jours de travail (vs {data.planned_duration} prévus). Un retard de {int(retard_estime)} jours est probable."
    elif retard_estime > 2:
        status = "warning"
        base_text = f"🟠 Attention pour {site_name_ref} : Le rythme actuel suggère un léger dépassement ({int(retard_estime)} jours) par rapport au planning."
    elif progression > 0.9:
        status = "good"
        base_text = f"✅ {site_name_ref} est en phase de finition. Les objectifs sont atteints."
    else:
        status = "good"
        base_text = f"✨ {site_name_ref} avance normalement. L'estimation IA ({int(predicted_total_days)}j) est alignée avec votre planning."

    # Ajouter recommandations météo si la localisation est fournie
    weather_note = ""
    if data.location and data.location.strip():
        try:
            coords = geocode_location(data.location)
            if coords:
                lat, lon = coords
                forecast = get_weather_forecast(lat, lon, days=3)  # Prévisions sur 3 jours
                
                if forecast and len(forecast) > 0:
                    # Analyser les conditions météo pour les prochains jours
                    today = forecast[0] if len(forecast) > 0 else None
                    tomorrow = forecast[1] if len(forecast) > 1 else None
                    
                    weather_advice = []
                    
                    if today:
                        if today.precipitation > 2:
                            weather_advice.append(f"Pluie prévue aujourd'hui ({today.precipitation:.1f}mm)")
                        elif today.precipitation > 0.5:
                            weather_advice.append(f"Risque de pluie aujourd'hui ({today.precipitation:.1f}mm)")
                        elif today.temperature < 5:
                            weather_advice.append(f"Température froide aujourd'hui ({today.temperature:.1f}°C)")
                        elif today.temperature > 30:
                            weather_advice.append(f"Température élevée aujourd'hui ({today.temperature:.1f}°C)")
                        else:
                            weather_advice.append(f"Conditions favorables aujourd'hui ({today.temperature:.1f}°C)")
                    
                    if tomorrow:
                        if tomorrow.precipitation > 2:
                            weather_advice.append(f"Pluie prévue demain ({tomorrow.precipitation:.1f}mm)")
                        elif tomorrow.precipitation > 0.5:
                            weather_advice.append(f"Risque de pluie demain ({tomorrow.precipitation:.1f}mm)")
                    
                    if weather_advice:
                        weather_note = f" 🌤️ Météo {data.location}: {' | '.join(weather_advice)}. Planifiez les tâches extérieures en conséquence."
                    else:
                        weather_note = f" 🌤️ Météo {data.location}: Conditions favorables pour les prochains jours. Idéal pour les travaux extérieurs."
                else:
                    weather_note = f" 🌤️ Consultez la météo de {data.location} pour optimiser les tâches extérieures."
            else:
                weather_note = f" 🌤️ Consultez la météo de {data.location} pour optimiser les tâches extérieures."
        except Exception as e:
            print(f"Erreur récupération météo pour résumé: {e}")
            weather_note = f" 🌤️ Consultez la météo de {data.location} pour optimiser les tâches extérieures."
    else:
        weather_note = " 🌤️ Consultez la météo pour optimiser les tâches extérieures."

    text = base_text + weather_note

    return SummaryResponse(summary=text, status=status, sites_mentioned=[data.site_name])


@app.post("/planning/optimize-weather", response_model=OptimizedPlanningResponse)
async def optimize_planning_with_weather(data: WeatherOptimizationInput):
    """Optimise le planning en tenant compte de la météo et des règles de métier"""
    location = data.location or "Paris"
    coords = geocode_location(location)
    
    if not coords:
        return OptimizedPlanningResponse(
            recommendations=[],
            best_dates=[],
            warnings=["Impossible de géolocaliser le chantier. Utilisation de Paris par défaut."]
        )
    
    lat, lon = coords
    forecast = get_weather_forecast(lat, lon, days=14)
    
    if not forecast:
        return OptimizedPlanningResponse(
            recommendations=[],
            best_dates=[],
            warnings=["Impossible de récupérer les prévisions météo."]
        )
    
    recommendations = []
    best_dates = []
    warnings = []
    
    # Analyser chaque tâche avec les prévisions
    for task in data.tasks:
        # Parser la date de la tâche (format YYYY-MM-DD)
        try:
            task_date = datetime.strptime(task.planned_date.split('T')[0], '%Y-%m-%d')
        except:
            task_date = datetime.now()
        
        # Trouver la prévision la plus proche
        closest_weather = None
        min_diff = float('inf')
        for weather in forecast:
            try:
                weather_date = datetime.strptime(weather.date, '%Y-%m-%d')
            except:
                continue
            diff = abs((task_date - weather_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_weather = weather
        
        if closest_weather:
            check = check_weather_for_role(task.task_role, closest_weather)
            recommendations.append(WeatherRecommendation(
                date=task.planned_date,
                temperature=closest_weather.temperature,
                precipitation=closest_weather.precipitation,
                favorable=check["favorable"],
                reason=check.get("reason", ""),
                recommendation=check.get("recommendation", "")
            ))
            
            if check["favorable"]:
                best_dates.append(task.planned_date)
            else:
                # Chercher une meilleure date dans les 14 prochains jours
                for weather in forecast:
                    try:
                        future_date = datetime.strptime(weather.date, '%Y-%m-%d')
                    except:
                        continue
                    if future_date > task_date:
                        future_check = check_weather_for_role(task.task_role, weather)
                        if future_check["favorable"]:
                            best_dates.append(weather.date)
                            warnings.append(
                                f"Tâche '{task.task_title}' : meilleure date le {weather.date} (au lieu de {task.planned_date})"
                            )
                            break
                # Si pas de meilleure date trouvée, garder la date originale
                if not any(d == task.planned_date.split('T')[0] for d in best_dates):
                    best_dates.append(task.planned_date.split('T')[0])
    
    return OptimizedPlanningResponse(
        recommendations=recommendations,
        best_dates=best_dates,
        warnings=warnings
    )


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
