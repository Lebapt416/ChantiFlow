#!/usr/bin/env python3
"""
Script de seeding pour peupler l'IA avec des données d'entraînement réalistes.
Envoie des données de feedback au modèle pour déclencher l'apprentissage automatique.

Usage:
    python seed_ai.py

Configuration:
    Modifiez la variable API_URL ci-dessous avec l'URL de votre API Railway.
    Exemple: https://chantiflow-production.up.railway.app
"""

import requests
import random
import time
import os
import sys

# ⚠️ URL à configurer : L'URL de ton API Railway
# Option 1: Définir directement ici
API_URL = os.getenv("ML_API_URL", "https://chantiflow-production.up.railway.app")

# Option 2: Lire depuis une variable d'environnement
# API_URL = os.getenv("ML_API_URL", "http://localhost:8000")

# Nettoyer l'URL (enlever le slash final si présent)
API_URL = API_URL.rstrip('/')

def generate_realistic_site():
    """Génère un chantier cohérent mathématiquement pour le BTP"""
    
    type_chantier = random.choice(['petit', 'moyen', 'gros', 'catastrophe'])
    
    if type_chantier == 'petit':
        nb_taches = random.randint(3, 15)
        complexite = random.uniform(1.0, 3.5)
        # Rapide
        duree = (nb_taches * 0.8) + (complexite * 1.5) + random.uniform(-1, 2)
        
    elif type_chantier == 'moyen':
        nb_taches = random.randint(15, 50)
        complexite = random.uniform(3.0, 7.0)
        # Standard
        duree = (nb_taches * 0.6) + (complexite * 3) + random.uniform(0, 5)
        
    elif type_chantier == 'gros':
        nb_taches = random.randint(50, 150)
        complexite = random.uniform(6.0, 9.5)
        # Effet d'échelle
        duree = (nb_taches * 0.5) + (complexite * 5) + random.uniform(5, 15)
        
    else: # 'catastrophe'
        nb_taches = random.randint(20, 60)
        complexite = random.uniform(8.0, 10.0)
        # Dérape complètement
        duree = (nb_taches * 1.2) + (complexite * 8) 

    return {
        "nombre_taches": int(nb_taches),
        "complexite": round(complexite, 1),
        "duree_reelle": round(max(1, duree), 1)
    }


def main():
    """Fonction principale de seeding"""
    print(f"🚀 Démarrage de l'entraînement intensif vers {API_URL}...")
    print(f"📡 Endpoint: {API_URL}/feedback\n")
    
    # Vérifier que l'API est accessible
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API accessible et opérationnelle\n")
        else:
            print(f"⚠️  API répond avec le code {health_response.status_code}")
            print("   Le seeding continuera quand même...\n")
    except requests.exceptions.RequestException as e:
        print(f"❌ Impossible de contacter l'API: {e}")
        print(f"   Vérifiez que l'URL est correcte: {API_URL}")
        sys.exit(1)
    
    success_count = 0
    error_count = 0
    
    # Envoi de 50 chantiers (Suffisant pour déclencher 10 entraînements)
    for i in range(50):
        data = generate_realistic_site()
        
        try:
            # On tape sur la route /feedback qui déclenche l'apprentissage
            response = requests.post(
                f"{API_URL}/feedback",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                status = response.json().get("status", "unknown")
                success_count += 1
                print(f"[{i+1}/50] ✅ Tâches: {data['nombre_taches']:3d} | "
                      f"Cplx: {data['complexite']:4.1f} -> {data['duree_reelle']:5.1f}j | "
                      f"Status: {status}")
            else:
                error_count += 1
                print(f"[{i+1}/50] ❌ Erreur API : {response.status_code} - {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            error_count += 1
            print(f"[{i+1}/50] ⏱️  Timeout - L'API met trop de temps à répondre")
        except requests.exceptions.RequestException as e:
            error_count += 1
            print(f"[{i+1}/50] ❌ Erreur connexion : {e}")
            # Continuer même en cas d'erreur pour voir combien passent
            
        time.sleep(0.1)  # Pause anti-spam
    
    print(f"\n{'='*60}")
    print(f"✅ Fin du seeding")
    print(f"   Succès: {success_count}/50")
    print(f"   Erreurs: {error_count}/50")
    print(f"   L'IA devrait avoir appris avec {success_count} nouveaux exemples !")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

