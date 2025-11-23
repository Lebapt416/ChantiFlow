/**
 * Fonctions pour la recommandation d'équipe via l'API FastAPI
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_PREDICTION_API_URL || 'http://localhost:8000';

type RecommandationEquipeInput = {
  type_tache: string;
  complexite: number;
};

type RecommandationEquipeResponse = {
  equipe_recommandee: string;
  performance_attendue: number;
};

/**
 * Recommande une équipe selon le type de tâche et la complexité
 * @param typeTache Type de tâche (ex: "maçonnerie", "plomberie")
 * @param complexite Complexité de la tâche (1-10)
 * @returns Équipe recommandée et performance attendue
 */
export async function recommanderEquipe(
  typeTache: string,
  complexite: number,
): Promise<RecommandationEquipeResponse> {
  if (!typeTache || typeTache.trim() === '') {
    throw new Error('Le type de tâche est requis.');
  }

  if (complexite < 1 || complexite > 10) {
    throw new Error('La complexité doit être entre 1 et 10.');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/recommander_equipe`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        type_tache: typeTache,
        complexite: complexite,
      } as RecommandationEquipeInput),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        `Erreur de l'API de recommandation d'équipe: ${response.status} - ${errorData.detail || response.statusText}`,
      );
    }

    const data: RecommandationEquipeResponse = await response.json();

    if (!data.equipe_recommandee || typeof data.performance_attendue !== 'number') {
      throw new Error('Réponse de l\'API de recommandation d\'équipe invalide.');
    }

    return data;
  } catch (error) {
    console.error('Erreur lors de la recommandation d\'équipe:', error);
    throw error;
  }
}

