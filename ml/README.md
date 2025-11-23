# Module de Prédiction de Durée de Chantier - ChantiFlow

Ce module contient l'IA de prédiction de durée de chantier utilisant PyTorch et FastAPI.

## Structure

- `train_model.py` - Script d'entraînement du modèle PyTorch
- `api.py` - API FastAPI pour servir le modèle
- `predictor.pt` - Modèle entraîné (généré après l'entraînement)
- `requirements.txt` - Dépendances Python

## Installation

1. Installer les dépendances Python :
```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Entraîner le modèle

```bash
python ml/train_model.py
```

Cela va :
- Générer 50 exemples de données factices
- Entraîner le modèle sur 100 epochs
- Sauvegarder le modèle dans `ml/predictor.pt`

### 2. Démarrer l'API

```bash
python ml/api.py
```

Ou avec uvicorn directement :
```bash
cd ml
python3 -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Note :** Utilisez `python3 -m uvicorn` au lieu de `uvicorn` directement pour éviter les erreurs de PATH.

L'API sera disponible sur `http://localhost:8000`

### 3. Utiliser depuis le front-end

Le module TypeScript `src/lib/ai/prediction.ts` est déjà configuré pour appeler l'API.

Exemple d'utilisation :
```typescript
import { getPrediction } from '@/lib/ai/prediction';

// Prédire la durée d'un chantier avec 20 tâches et complexité 5.5
const duree = await getPrediction(20, 5.5);
console.log(`Durée estimée: ${duree} jours`);
```

## Endpoints API

- `GET /` - Informations sur l'API
- `GET /health` - Vérification de santé
- `POST /predict` - Prédiction de durée

### Exemple de requête POST /predict

```json
{
  "nombre_taches": 20,
  "complexite": 5.5
}
```

### Exemple de réponse

```json
{
  "duree_estimee": 25.5
}
```

## Configuration

Pour changer l'URL de l'API depuis le front-end, définir la variable d'environnement :
```
NEXT_PUBLIC_PREDICTION_API_URL=http://localhost:8000
```

## Notes

- Le modèle actuel utilise des données factices pour l'entraînement
- En production, remplacer par de vraies données historiques de chantiers
- Les valeurs de normalisation sont approximatives et devraient être sauvegardées avec le modèle

