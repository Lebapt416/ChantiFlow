'use server';

import { calculateDaysNeeded, MAX_WORKING_HOURS_PER_DAY } from './work-rules';

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

type PlanningResult = {
  orderedTasks: Array<{
    taskId: string;
    order: number;
    startDate: string;
    endDate: string;
    assignedWorkerId?: string | null; // Ancien format (compatibilité)
    assignedWorkerIds?: string[]; // Nouveau format (collaboration)
    dependencies: string[];
    priority: 'high' | 'medium' | 'low';
    estimatedHours?: number;
  }>;
  newDeadline?: string;
  warnings: string[];
  reasoning: string;
};

/**
 * Génère un planning intelligent avec Google Gemini
 * Utilise l'API Google Gemini pour analyser et optimiser le planning
 */
export async function generateAIPlanning(
  tasks: Task[],
  workers: Worker[],
  deadline: string | null,
  siteName: string,
  siteId?: string,
): Promise<PlanningResult> {
  const { isGeminiConfigured, generateWithGemini } = await import('./gemini');

  // Si pas d'API key Gemini, utiliser l'IA locale avancée
  if (!(await isGeminiConfigured())) {
    console.log('[AI Planning] Pas d\'API key Google Gemini, utilisation de l\'IA locale');
    const { generateLocalAIPlanning } = await import('./local-planning');
    const planning = await generateLocalAIPlanning(tasks, workers, deadline, siteName);
    
    // Enregistrer pour l'entraînement même pour l'IA locale
    if (siteId) {
      const { savePlanningResult } = await import('./training');
      await savePlanningResult(siteId, tasks, workers, planning, siteName, deadline);
    }
    
    return planning;
  }

  try {
    console.log('[AI Planning] Appel Google Gemini avec', tasks.length, 'tâches');
    // Préparer le prompt pour Gemini
    const tasksDescription = tasks
      .map(
        (task, index) =>
          `${index + 1}. ${task.title} (${task.required_role || 'Rôle libre'}, ${task.duration_hours || '?'}h)`,
      )
      .join('\n');

    const workersDescription =
      workers.length > 0
        ? workers.map((w) => `- ${w.name} (${w.role || 'Rôle non défini'})`).join('\n')
        : 'Aucun employé assigné';

    const prompt = `Tu es un expert en gestion de chantiers de construction en France. Analyse ce chantier et génère un planning optimisé en respectant STRICTEMENT les lois du travail françaises.

Chantier: ${siteName}
Deadline: ${deadline ? new Date(deadline).toLocaleDateString('fr-FR') : 'Non définie'}

Tâches à planifier:
${tasksDescription}

Équipe disponible:
${workersDescription}

LOIS DU TRAVAIL FRANÇAISES À RESPECTER STRICTEMENT:
- Maximum 8h de travail EFFECTIF par jour (hors pause déjeuner)
- Pause déjeuner OBLIGATOIRE de 1h (12h-13h généralement)
- Personne ne travaille 12h par jour - c'est ILLÉGAL et irréaliste
- Si une tâche dépasse 8h, elle DOIT être répartie sur plusieurs jours
- Pause obligatoire de 20 minutes après 6h de travail (incluse dans la pause déjeuner si > 6h)
- Repos minimum de 11h entre deux journées
- Repos hebdomadaire de 24h consécutives (généralement dimanche)
- Jours fériés exclus du planning

EXEMPLE: Une tâche de 12h doit être répartie sur 2 jours (8h le jour 1, 4h le jour 2)

RÈGLES DE COLLABORATION:
- Les workers du même métier peuvent collaborer sur une même tâche
- Si une tâche dure plus de 8h, assigne plusieurs workers du même métier si disponibles
- Prends en compte le temps estimé (duration_hours) de chaque tâche

RECONNAISSANCE DES MÉTIERS:
- Maçon: fondation, structure, mur, cloison, parpaing, brique, ciment
- Électricien: électricité, câblage, tableau, prise, éclairage
- Plombier: plomberie, sanitaire, eau, chauffage
- Peintre: peinture, finition, enduit, revêtement
- Carreleur: carrelage, faïence, sol
- Charpentier: charpente, bois, toiture, ossature
- Couvreur: couverture, toiture, tuile, ardoise
- Menuisier: menuiserie, fenêtre, porte, parquet

Génère un planning JSON avec:
1. Ordre logique des tâches (dépendances, séquence optimale)
2. Dates de début/fin réalistes en respectant les lois du travail
3. Assignation des workers (peut être un tableau pour collaboration)
4. Priorités (high/medium/low)
5. Temps estimé de chaque tâche
6. Une explication détaillée de ton raisonnement

Réponds UNIQUEMENT avec un JSON valide dans ce format:
{
  "orderedTasks": [
    {
      "taskId": "id de la tâche",
      "order": 1,
      "startDate": "YYYY-MM-DD",
      "endDate": "YYYY-MM-DD",
      "assignedWorkerIds": ["id1", "id2"] ou ["id1"] ou [],
      "dependencies": ["id1", "id2"],
      "priority": "high|medium|low",
      "estimatedHours": nombre d'heures
    }
  ],
  "warnings": ["avertissement 1", "avertissement 2"],
  "reasoning": "Explication détaillée de ton analyse, des lois respectées, et des choix de collaboration"
}`;

    const systemInstruction = 'Tu es un expert en gestion de projets de construction en France. Tu génères des plannings optimisés en JSON en respectant STRICTEMENT les lois du travail françaises (Code du travail). Tu reconnais les métiers et permets la collaboration entre workers du même métier. Tu prends en compte le temps estimé (duration_hours) de chaque tâche. Réponds UNIQUEMENT en JSON valide.';

    const content = await generateWithGemini(prompt, systemInstruction, {
      temperature: 0.7,
      maxOutputTokens: 4000,
      responseFormat: 'json',
    });

    if (!content) {
      console.error('[AI Planning] Réponse Gemini vide');
      throw new Error('Réponse Gemini vide');
    }

    console.log('[AI Planning] Réponse Gemini reçue, longueur:', content.length);

    let planning: PlanningResult;
    try {
      planning = JSON.parse(content) as PlanningResult;
    } catch (parseError) {
      console.error('[AI Planning] Erreur parsing JSON:', parseError);
      console.error('[AI Planning] Contenu reçu:', content.substring(0, 500));
      throw new Error('Erreur lors du parsing de la réponse Gemini');
    }

    // Valider et compléter les dates en respectant les lois du travail
    const startDate = new Date();
    let currentDate = new Date(startDate);
    
    planning.orderedTasks = planning.orderedTasks.map((task) => {
      const taskObj = tasks.find((t) => t.id === task.taskId);
      const duration = taskObj?.duration_hours || task.estimatedHours || 8;
      
      // Utiliser la durée estimée de la tâche
      const estimatedHours = task.estimatedHours || duration;
      
      // Calculer les jours en respectant les lois (8h/jour max avec pause déjeuner)
      // Si une tâche dépasse 8h, elle doit être répartie sur plusieurs jours
      const daysNeeded = calculateDaysNeeded(estimatedHours, MAX_WORKING_HOURS_PER_DAY);

      // La date de début est la date actuelle (ou la date de fin de la tâche précédente)
      const taskStartDate = new Date(currentDate);
      
      // La date de fin est calculée en fonction du nombre de jours nécessaires
      const taskEndDate = new Date(taskStartDate);
      taskEndDate.setDate(taskEndDate.getDate() + daysNeeded - 1); // -1 car le premier jour compte
      
      // Avancer la date courante pour la prochaine tâche
      currentDate = new Date(taskEndDate);
      currentDate.setDate(currentDate.getDate() + 1); // Commencer le jour suivant
      
      // Normaliser les assignedWorkerIds si on a l'ancien format
      const assignedWorkerIds = task.assignedWorkerIds || 
                                (task.assignedWorkerId ? [task.assignedWorkerId] : []);

      return {
        ...task,
        startDate: taskStartDate.toISOString().split('T')[0],
        endDate: taskEndDate.toISOString().split('T')[0],
        assignedWorkerIds,
        estimatedHours, // Garder la valeur réelle pour la distribution sur plusieurs jours
      };
    });

    // Valider que le planning contient les bonnes données
    if (!planning.orderedTasks || !Array.isArray(planning.orderedTasks)) {
      console.error('[AI Planning] Format de planning invalide');
      throw new Error('Format de planning invalide depuis Gemini');
    }

    console.log('[AI Planning] Planning généré avec succès:', planning.orderedTasks.length, 'tâches');
    
    // Enregistrer pour l'entraînement continu
    if (siteId) {
      try {
        const { savePlanningForTraining } = await import('./improved-planning');
        // Adapter le format pour la compatibilité
        const adaptedPlanning = {
          ...planning,
          orderedTasks: planning.orderedTasks.map(task => ({
            ...task,
            assignedWorkerIds: task.assignedWorkerIds || (task.assignedWorkerId ? [task.assignedWorkerId] : []),
            estimatedHours: task.estimatedHours || 8,
          })),
        };
        await savePlanningForTraining(siteId, tasks, workers, adaptedPlanning, siteName, deadline);
      } catch (error) {
        console.error('[AI Training] Error saving planning for training:', error);
        // Ne pas bloquer si l'enregistrement échoue
      }
    }
    
    return planning;
  } catch (error) {
    // En cas d'erreur, utiliser l'IA locale avancée
    console.error('[AI Planning] Erreur Gemini, fallback IA locale:', error);
    const { generateLocalAIPlanning } = await import('./local-planning');
    const localPlanning = await generateLocalAIPlanning(tasks, workers, deadline, siteName);
    localPlanning.reasoning = `⚠️ Google Gemini non disponible (${error instanceof Error ? error.message : 'Erreur inconnue'}). ${localPlanning.reasoning}`;
    return localPlanning;
  }
}


