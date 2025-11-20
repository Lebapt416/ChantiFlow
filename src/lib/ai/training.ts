'use server';

import { createSupabaseServerClient } from '@/lib/supabase/server';

type PlanningResult = {
  orderedTasks: Array<{
    taskId: string;
    order: number;
    startDate: string;
    endDate: string;
    assignedWorkerId: string | null;
    priority: 'high' | 'medium' | 'low';
  }>;
  warnings: string[];
  reasoning: string;
};

type Task = {
  id: string;
  title: string;
  required_role: string | null;
  duration_hours: number | null;
  status: 'pending' | 'done';
};

type Worker = {
  id: string;
  name: string;
  email: string;
  role: string | null;
};

/**
 * Enregistre un résultat de planning pour l'entraînement de l'IA
 */
export async function savePlanningResult(
  siteId: string,
  tasks: Task[],
  workers: Worker[],
  planning: PlanningResult,
  siteName: string,
  deadline: string | null,
) {
  try {
    const supabase = await createSupabaseServerClient();
    
    // Stocker le résultat dans une table d'entraînement (à créer dans Supabase)
    // Pour l'instant, on utilise la table reports ou on crée une nouvelle logique
    
    // On peut aussi stocker dans localStorage côté client ou dans une table dédiée
    // Ici, on va simplement logger pour l'instant et améliorer l'algorithme local
    
    console.log('[AI Training] Planning saved:', {
      siteId,
      siteName,
      taskCount: tasks.length,
      workerCount: workers.length,
      planningTasks: planning.orderedTasks.length,
      warnings: planning.warnings.length,
    });
    
    return { success: true };
  } catch (error) {
    console.error('[AI Training] Error saving planning:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

/**
 * Récupère les patterns d'apprentissage depuis les plannings précédents
 */
export async function getLearningPatterns() {
  try {
    const supabase = await createSupabaseServerClient();
    
    // Récupérer les plannings précédents pour analyser les patterns
    // Pour l'instant, on retourne des patterns par défaut
    
    return {
      commonDependencies: [
        { from: 'fondation', to: 'structure', frequency: 0.9 },
        { from: 'structure', to: 'électricité', frequency: 0.8 },
        { from: 'structure', to: 'plomberie', frequency: 0.8 },
        { from: 'électricité', to: 'peinture', frequency: 0.7 },
        { from: 'plomberie', to: 'carrelage', frequency: 0.7 },
      ],
      roleAssignments: [
        { role: 'maçon', tasks: ['fondation', 'structure', 'mur'], frequency: 0.85 },
        { role: 'électricien', tasks: ['électricité', 'câblage'], frequency: 0.9 },
        { role: 'plombier', tasks: ['plomberie', 'sanitaire'], frequency: 0.9 },
        { role: 'peintre', tasks: ['peinture', 'finition'], frequency: 0.8 },
      ],
      averageDurations: {
        fondation: 16,
        structure: 40,
        électricité: 24,
        plomberie: 24,
        peinture: 16,
        carrelage: 20,
      },
    };
  } catch (error) {
    console.error('[AI Training] Error getting patterns:', error);
    return null;
  }
}

/**
 * Améliore l'algorithme local en utilisant les patterns appris
 */
export async function enhanceLocalPlanningWithLearning(
  tasks: Task[],
  workers: Worker[],
  deadline: string | null,
  siteName: string,
) {
  const patterns = await getLearningPatterns();
  
  if (!patterns) {
    return null; // Utiliser l'algorithme de base
  }
  
  // Utiliser les patterns pour améliorer l'analyse des dépendances
  // et l'assignation des workers
  
  // Cette fonction sera appelée depuis local-planning.ts pour améliorer les résultats
  
  return patterns;
}

