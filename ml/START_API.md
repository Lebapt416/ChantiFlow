# Guide de démarrage de l'API FastAPI

## Prérequis

- Python 3.9 ou supérieur
- pip (gestionnaire de paquets Python)

## Installation

### 1. Installer les dépendances Python

Depuis la racine du projet :

```bash
cd ml
pip3 install -r requirements.txt
```

Ou si vous utilisez un environnement virtuel (recommandé) :

```bash
cd ml
python3 -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate  # Sur Windows
pip install -r requirements.txt
```

### 2. Vérifier que le modèle est entraîné

Si le fichier `predictor.pt` n'existe pas, entraînez d'abord le modèle :

```bash
cd ml
python3 train_model.py
```

Cela créera le fichier `predictor.pt` nécessaire pour l'API.

## Démarrage de l'API

### Option 1 : Avec le script Python (simple)

Depuis le dossier `ml` :

```bash
cd ml
python3 api.py
```

### Option 2 : Avec uvicorn directement (recommandé pour le développement)

Depuis le dossier `ml` :

```bash
cd ml
python3 -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Note :** Utilisez `python3 -m uvicorn` au lieu de `uvicorn` directement pour éviter les erreurs de PATH.

L'option `--reload` permet de recharger automatiquement l'API lors des modifications.

## Vérification

Une fois l'API démarrée, vous devriez voir :

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
✅ Modèle chargé depuis ml/predictor.pt
📊 Modèle prêt à faire des prédictions
INFO:     Application startup complete.
```

Vous pouvez vérifier que l'API fonctionne en visitant :
- `http://localhost:8000/` - Page d'accueil
- `http://localhost:8000/health` - Vérification de santé
- `http://localhost:8000/docs` - Documentation interactive Swagger

## Dépannage

### Erreur : "command not found: uvicorn"

Solution : Installez les dépendances :
```bash
cd ml
pip3 install -r requirements.txt
```

### Erreur : "Le modèle predictor.pt n'existe pas"

Solution : Entraînez d'abord le modèle :
```bash
cd ml
python3 train_model.py
```

### Erreur : "Module not found"

Solution : Assurez-vous d'être dans le bon dossier et que les dépendances sont installées :
```bash
cd ml
pip3 install -r requirements.txt
```

### Le port 8000 est déjà utilisé

Solution : Changez le port dans la commande :
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8001
```

Et mettez à jour la variable d'environnement `NEXT_PUBLIC_PREDICTION_API_URL` dans votre fichier `.env.local`.

