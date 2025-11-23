"""
Module d'entraînement du modèle de prédiction de durée de chantier
Utilise PyTorch pour créer et entraîner un réseau de neurones simple
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path


class ChantierPredictor(nn.Module):
    """
    Réseau de neurones simple pour prédire la durée d'un chantier
    Architecture: 2 entrées (nombre_taches, complexite) -> 1 sortie (duree)
    """
    
    def __init__(self):
        super(ChantierPredictor, self).__init__()
        # Couche linéaire: 2 entrées -> 1 sortie
        self.fc = nn.Linear(2, 1)
    
    def forward(self, x):
        """
        Passe avant du réseau
        Args:
            x: Tenseur de forme (batch_size, 2) contenant [nombre_taches, complexite]
        Returns:
            Tenseur de forme (batch_size, 1) contenant la durée prédite
        """
        return self.fc(x)


def generate_fake_data(n_samples=50):
    """
    Génère des données factices pour l'entraînement
    Args:
        n_samples: Nombre d'exemples à générer
    Returns:
        tuple: (X, y) où X est un tenseur (n_samples, 2) et y est un tenseur (n_samples, 1)
    """
    np.random.seed(42)  # Pour la reproductibilité
    
    # Générer des données réalistes
    nombre_taches = np.random.randint(5, 50, size=(n_samples, 1))
    complexite = np.random.uniform(1.0, 10.0, size=(n_samples, 1))
    
    # Relation: durée = base + (nombre_taches * coef_taches) + (complexite * coef_complexite) + bruit
    # Formule réaliste: durée en jours = nombre_taches * 0.5 + complexite * 2 + bruit
    duree = (nombre_taches * 0.5) + (complexite * 2) + np.random.normal(0, 2, size=(n_samples, 1))
    duree = np.maximum(duree, 1.0)  # Durée minimum de 1 jour
    
    # Normaliser les entrées pour améliorer l'entraînement
    X = np.hstack([nombre_taches, complexite])
    X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Convertir en tenseurs PyTorch
    X_tensor = torch.FloatTensor(X_normalized)
    y_tensor = torch.FloatTensor(duree)
    
    return X_tensor, y_tensor


def train_model():
    """
    Fonction principale d'entraînement du modèle
    """
    print("🚀 Démarrage de l'entraînement du modèle ChantierPredictor...")
    
    # Créer le modèle
    model = ChantierPredictor()
    
    # Générer les données d'entraînement
    X_train, y_train = generate_fake_data(n_samples=50)
    print(f"✅ Données générées: {X_train.shape[0]} exemples")
    
    # Définir la fonction de perte et l'optimiseur
    criterion = nn.MSELoss()  # Mean Squared Error
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent
    
    # Boucle d'entraînement
    n_epochs = 100
    print(f"📊 Entraînement sur {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        # Remettre les gradients à zéro
        optimizer.zero_grad()
        
        # Passe avant
        predictions = model(X_train)
        
        # Calculer la perte
        loss = criterion(predictions, y_train)
        
        # Passe arrière
        loss.backward()
        
        # Mettre à jour les poids
        optimizer.step()
        
        # Afficher la progression tous les 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
    
    # Sauvegarder le modèle dans le répertoire courant
    model_path = Path(__file__).parent / "predictor.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'ChantierPredictor',
    }, model_path)
    
    print(f"✅ Modèle sauvegardé dans {model_path}")
    
    # Afficher quelques prédictions pour vérification
    model.eval()
    with torch.no_grad():
        sample_predictions = model(X_train[:5])
        print("\n📈 Exemples de prédictions:")
        for i in range(5):
            print(f"  Exemple {i+1}: {X_train[i].numpy()} -> Prédit: {sample_predictions[i].item():.2f} jours, Réel: {y_train[i].item():.2f} jours")
    
    print("\n🎉 Entraînement terminé avec succès!")


if __name__ == "__main__":
    train_model()

