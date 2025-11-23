/**
 * Fonctions pour l'analyse de risque de retard via l'API FastAPI
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_PREDICTION_API_URL || 'http://localhost:8000';

type AnalyseRisqueInput = {
  historique_taches_similaires: number[];
  complexite_actuelle: number;
};

type AnalyseRisqueResponse = {
  risque_pourcentage: number;
  justification: string;
};

/**
 * Analyse le risque de retard d'un chantier basé sur l'historique et la complexité
 * @param historique Liste des durées réelles des tâches similaires (en jours)
 * @param complexite Complexité actuelle du chantier (1-10)
 * @returns Pourcentage de risque et justification
 */
export async function analyserRisqueRetard(
  historique: number[],
  complexite: number,
): Promise<AnalyseRisqueResponse> {
  if (complexite < 1 || complexite > 10) {
    throw new Error('La complexité doit être entre 1 et 10.');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/analyse_risque`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        historique_taches_similaires: historique,
        complexite_actuelle: complexite,
      } as AnalyseRisqueInput),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        `Erreur de l'API d'analyse de risque: ${response.status} - ${errorData.detail || response.statusText}`,
      );
    }

    const data: AnalyseRisqueResponse = await response.json();

    if (typeof data.risque_pourcentage !== 'number' || !data.justification) {
      throw new Error('Réponse de l\'API d\'analyse de risque invalide.');
    }

    return data;
  } catch (error) {
    console.error('Erreur lors de l\'analyse de risque:', error);
    
    // Gérer les erreurs de réseau spécifiquement
    if (error instanceof TypeError && (error.message.includes('fetch') || error.message.includes('Failed to fetch'))) {
      throw new Error(
        'Impossible de se connecter à l\'API d\'analyse de risque. ' +
        'Vérifiez que le serveur FastAPI est démarré (port 8000) et que l\'URL est correcte.'
      );
    }
    
    // Si c'est déjà une Error avec un message, la relancer
    if (error instanceof Error) {
      throw error;
    }
    
    // Sinon, créer une nouvelle erreur
    throw new Error('Erreur inconnue lors de l\'analyse de risque.');
  }
}

