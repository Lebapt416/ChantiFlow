/**
 * Module front-end pour appeler l'API de prédiction de durée de chantier
 * Utilise fetch API pour communiquer avec le backend FastAPI
 */

/**
 * Interface pour les données d'entrée de prédiction
 */
export interface ChantierInput {
  nombre_taches: number;
  complexite: number;
}

/**
 * Interface pour la réponse de prédiction
 */
export interface ChantierPrediction {
  duree_estimee: number;
}

/**
 * Configuration de l'API
 * En production, remplacer par l'URL réelle de l'API
 */
const API_BASE_URL = process.env.NEXT_PUBLIC_PREDICTION_API_URL || 'http://localhost:8000';

/**
 * Fonction asynchrone pour obtenir une prédiction de durée de chantier
 * 
 * @param taches - Nombre de tâches du chantier (entre 1 et 100)
 * @param complexite - Niveau de complexité du chantier (entre 1.0 et 10.0)
 * @returns Promise<number> - Durée estimée en jours
 * @throws Error si la requête échoue
 * 
 * @example
 * ```typescript
 * try {
 *   const duree = await getPrediction(20, 5.5);
 *   console.log(`Durée estimée: ${duree} jours`);
 * } catch (error) {
 *   console.error('Erreur:', error);
 * }
 * ```
 */
export async function getPrediction(
  taches: number,
  complexite: number
): Promise<number> {
  // Validation des entrées
  if (taches < 1 || taches > 100) {
    throw new Error('Le nombre de tâches doit être entre 1 et 100');
  }
  
  if (complexite < 1.0 || complexite > 10.0) {
    throw new Error('La complexité doit être entre 1.0 et 10.0');
  }

  try {
    // Préparer les données à envoyer
    const inputData: ChantierInput = {
      nombre_taches: Math.round(taches),
      complexite: Math.round(complexite * 10) / 10, // Arrondir à 1 décimale
    };

    // Faire l'appel API
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(inputData),
    });

    // Vérifier si la requête a réussi
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || 
        `Erreur HTTP: ${response.status} ${response.statusText}`
      );
    }

    // Parser la réponse JSON
    const prediction: ChantierPrediction = await response.json();

    // Retourner la durée estimée
    return prediction.duree_estimee;

  } catch (error) {
    // Gérer les erreurs de réseau
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error(
        'Impossible de se connecter à l\'API de prédiction. ' +
        'Vérifiez que le serveur est démarré et que l\'URL est correcte.'
      );
    }

    // Re-lancer les autres erreurs
    throw error;
  }
}

/**
 * Fonction utilitaire pour vérifier si l'API est disponible
 * 
 * @returns Promise<boolean> - true si l'API est disponible, false sinon
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Fonction pour obtenir les informations de l'API
 * 
 * @returns Promise<object> - Informations sur l'API
 */
export async function getApiInfo(): Promise<{ message: string; status: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Erreur HTTP: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    throw new Error(`Impossible de récupérer les informations de l'API: ${error}`);
  }
}

