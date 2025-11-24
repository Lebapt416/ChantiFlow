# 🌱 Script de Seeding pour l'IA

Ce script permet de peupler l'IA avec des données d'entraînement réalistes pour améliorer les prédictions.

## 📋 Prérequis

```bash
pip install requests
```

## 🚀 Utilisation

### 1. Configurer l'URL de l'API

Modifiez la variable `API_URL` dans `seed_ai.py` avec l'URL de votre API Railway :

```python
API_URL = "https://votre-api.up.railway.app"
```

Ou utilisez une variable d'environnement :

```bash
export ML_API_URL="https://votre-api.up.railway.app"
python seed_ai.py
```

### 2. Exécuter le script

```bash
python seed_ai.py
```

## 📊 Ce que fait le script

- Génère **50 chantiers réalistes** avec différentes tailles :
  - **Petit** : 3-15 tâches, complexité 1.0-3.5
  - **Moyen** : 15-50 tâches, complexité 3.0-7.0
  - **Gros** : 50-150 tâches, complexité 6.0-9.5
  - **Catastrophe** : 20-60 tâches, complexité 8.0-10.0 (retards importants)

- Envoie chaque chantier à l'endpoint `/feedback` de l'API
- Déclenche automatiquement l'entraînement tous les 5 nouveaux échantillons
- Affiche le statut de chaque envoi

## 🎯 Résultat attendu

Après l'exécution, l'IA aura :
- **10 cycles d'entraînement** déclenchés (50 échantillons ÷ 5)
- Des prédictions plus précises basées sur des données réalistes
- Une meilleure compréhension des différents types de chantiers

## ⚠️ Notes

- Le script vérifie d'abord que l'API est accessible via `/health`
- En cas d'erreur, le script continue pour maximiser les données envoyées
- Une pause de 0.1s entre chaque envoi pour éviter le spam

