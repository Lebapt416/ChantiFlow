'use server';

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
 * Génère un planning intelligent avec OpenAI
 * Utilise l'API OpenAI pour analyser et optimiser le planning
 */
export async function generateAIPlanning(
  tasks: Task[],
  workers: Worker[],
  deadline: string | null,
  siteName: string,
  siteId?: string,
): Promise<PlanningResult> {
  const apiKey = process.env.OPENAI_API_KEY;

  // Si pas d'API key, utiliser l'IA locale avancée
  if (!apiKey || apiKey.trim() === '') {
    console.log('[AI Planning] Pas d\'API key OpenAI, utilisation de l\'IA locale');
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
    console.log('[AI Planning] Appel OpenAI avec', tasks.length, 'tâches');
    // Préparer le prompt pour OpenAI
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

LOIS DU TRAVAIL FRANÇAISES À RESPECTER:
- Maximum 10h/jour (avec dérogation) ou 8h/jour (standard)
- Pause obligatoire de 20 minutes après 6h de travail
- Repos minimum de 11h entre deux journées
- Repos hebdomadaire de 24h consécutives (généralement dimanche)
- Jours fériés exclus du planning

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

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content:
              'Tu es un expert en gestion de projets de construction en France. Tu génères des plannings optimisés en JSON en respectant STRICTEMENT les lois du travail françaises (Code du travail). Tu reconnais les métiers et permets la collaboration entre workers du même métier. Tu prends en compte le temps estimé (duration_hours) de chaque tâche. Réponds UNIQUEMENT en JSON valide.',
          },
          {
            role: 'user',
            content: prompt,
          },
        ],
        temperature: 0.7,
        response_format: { type: 'json_object' },
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[AI Planning] Erreur OpenAI:', response.status, errorText);
      
      // Gestion spécifique de l'erreur 429 (rate limit)
      if (response.status === 429) {
        throw new Error(
          'Quota OpenAI dépassé (429). Vous avez fait trop de requêtes. Attendez quelques minutes ou vérifiez votre quota sur platform.openai.com/usage',
        );
      }
      
      // Gestion de l'erreur 401 (clé invalide)
      if (response.status === 401) {
        throw new Error('Clé API OpenAI invalide. Vérifiez votre clé sur platform.openai.com/api-keys');
      }
      
      throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    const content = data.choices[0]?.message?.content;

    if (!content) {
      console.error('[AI Planning] Réponse OpenAI vide:', data);
      throw new Error('Réponse OpenAI vide');
    }

    console.log('[AI Planning] Réponse OpenAI reçue, longueur:', content.length);

    let planning: PlanningResult;
    try {
      planning = JSON.parse(content) as PlanningResult;
    } catch (parseError) {
      console.error('[AI Planning] Erreur parsing JSON:', parseError);
      console.error('[AI Planning] Contenu reçu:', content.substring(0, 500));
      throw new Error('Erreur lors du parsing de la réponse OpenAI');
    }

    // Valider et compléter les dates en respectant les lois du travail
    const startDate = new Date();
    planning.orderedTasks = planning.orderedTasks.map((task, index) => {
      const taskObj = tasks.find((t) => t.id === task.taskId);
      const duration = taskObj?.duration_hours || task.estimatedHours || 8;
      
      // Utiliser la durée estimée de la tâche
      const estimatedHours = task.estimatedHours || duration;
      
      // Calculer les jours en respectant les lois (8h/jour max, pauses, repos)
      const workingHoursPerDay = 8;
      const daysNeeded = Math.ceil(estimatedHours / workingHoursPerDay);

      const taskStartDate = new Date(startDate);
      taskStartDate.setDate(taskStartDate.getDate() + index);

      const taskEndDate = new Date(taskStartDate);
      taskEndDate.setDate(taskEndDate.getDate() + daysNeeded);
      
      // Normaliser les assignedWorkerIds si on a l'ancien format
      const assignedWorkerIds = task.assignedWorkerIds || 
                                (task.assignedWorkerId ? [task.assignedWorkerId] : []);

      return {
        ...task,
        startDate: taskStartDate.toISOString().split('T')[0],
        endDate: taskEndDate.toISOString().split('T')[0],
        assignedWorkerIds,
        estimatedHours,
      };
    });

    // Valider que le planning contient les bonnes données
    if (!planning.orderedTasks || !Array.isArray(planning.orderedTasks)) {
      console.error('[AI Planning] Format de planning invalide');
      throw new Error('Format de planning invalide depuis OpenAI');
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
        await savePlanningForTraining(siteId, tasks, workers, adaptedPlanning as any, siteName, deadline);
      } catch (error) {
        console.error('[AI Training] Error saving planning for training:', error);
        // Ne pas bloquer si l'enregistrement échoue
      }
    }
    
    return planning;
  } catch (error) {
    // En cas d'erreur, utiliser l'IA locale avancée
    console.error('[AI Planning] Erreur OpenAI, fallback IA locale:', error);
    const { generateLocalAIPlanning } = await import('./local-planning');
    const localPlanning = await generateLocalAIPlanning(tasks, workers, deadline, siteName);
    localPlanning.reasoning = `⚠️ OpenAI non disponible (${error instanceof Error ? error.message : 'Erreur inconnue'}). ${localPlanning.reasoning}`;
    return localPlanning;
  }
}

/**
 * Version basique sans IA (fallback ultime - ne devrait plus être utilisé)
 */
function generateBasicPlanning(
  tasks: Task[],
  workers: Worker[],
  deadline: string | null,
): PlanningResult {
  const startDate = new Date();
  const sortedTasks = [...tasks].sort((a, b) => {
    // Tâches de préparation en premier
    const prepKeywords = ['fondation', 'structure', 'terrassement'];
    const aIsPrep = prepKeywords.some((k) => a.title.toLowerCase().includes(k));
    const bIsPrep = prepKeywords.some((k) => b.title.toLowerCase().includes(k));
    if (aIsPrep && !bIsPrep) return -1;
    if (!aIsPrep && bIsPrep) return 1;

    // Tâches de finition en dernier
    const finishKeywords = ['peinture', 'finition', 'nettoyage'];
    const aIsFinish = finishKeywords.some((k) => a.title.toLowerCase().includes(k));
    const bIsFinish = finishKeywords.some((k) => b.title.toLowerCase().includes(k));
    if (aIsFinish && !bIsFinish) return 1;
    if (!aIsFinish && bIsFinish) return -1;

    return 0;
  });

  const orderedTasks = sortedTasks.map((task, index) => {
    const duration = task.duration_hours || 8;
    const daysNeeded = Math.ceil(duration / 8);

    const taskStartDate = new Date(startDate);
    taskStartDate.setDate(taskStartDate.getDate() + index);

    const taskEndDate = new Date(taskStartDate);
    taskEndDate.setDate(taskEndDate.getDate() + daysNeeded);

    const requiredRole = task.required_role;
    const assignedWorker = requiredRole
      ? workers.find((w) => w.role?.toLowerCase() === requiredRole.toLowerCase())
      : workers[0] || null;

    const priority: 'high' | 'medium' | 'low' =
      index === 0 ? 'high' : index >= sortedTasks.length - 2 ? 'low' : 'medium';

    return {
      taskId: task.id,
      order: index + 1,
      startDate: taskStartDate.toISOString().split('T')[0],
      endDate: taskEndDate.toISOString().split('T')[0],
      assignedWorkerId: assignedWorker?.id || null,
      dependencies: [],
      priority,
    };
  });

  return {
    orderedTasks,
    warnings: [],
    reasoning: 'Planning généré avec un algorithme de base (sans IA). Pour une analyse plus poussée, configurez OPENAI_API_KEY.',
  };
}

