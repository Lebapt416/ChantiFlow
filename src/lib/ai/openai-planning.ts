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
    assignedWorkerId: string | null;
    dependencies: string[];
    priority: 'high' | 'medium' | 'low';
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
): Promise<PlanningResult> {
  const apiKey = process.env.OPENAI_API_KEY;

  // Si pas d'API key, utiliser la version basique
  if (!apiKey || apiKey.trim() === '') {
    console.log('[AI Planning] Pas d\'API key OpenAI, utilisation du fallback basique');
    return generateBasicPlanning(tasks, workers, deadline);
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

    const prompt = `Tu es un expert en gestion de chantiers de construction. Analyse ce chantier et génère un planning optimisé.

Chantier: ${siteName}
Deadline: ${deadline ? new Date(deadline).toLocaleDateString('fr-FR') : 'Non définie'}

Tâches à planifier:
${tasksDescription}

Équipe disponible:
${workersDescription}

Génère un planning JSON avec:
1. Ordre logique des tâches (dépendances, séquence optimale)
2. Dates de début/fin réalistes
3. Assignation des workers selon les rôles
4. Priorités (high/medium/low)
5. Une explication de ton raisonnement

Réponds UNIQUEMENT avec un JSON valide dans ce format:
{
  "orderedTasks": [
    {
      "taskId": "id de la tâche",
      "order": 1,
      "startDate": "YYYY-MM-DD",
      "endDate": "YYYY-MM-DD",
      "assignedWorkerId": "id du worker ou null",
      "dependencies": ["id1", "id2"],
      "priority": "high|medium|low"
    }
  ],
  "warnings": ["avertissement 1", "avertissement 2"],
  "reasoning": "Explication de ton analyse et de tes choix"
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
              'Tu es un expert en gestion de projets de construction. Tu génères des plannings optimisés en JSON. Réponds UNIQUEMENT en JSON valide.',
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

    // Valider et compléter les dates
    const startDate = new Date();
    planning.orderedTasks = planning.orderedTasks.map((task, index) => {
      const taskObj = tasks.find((t) => t.id === task.taskId);
      const duration = taskObj?.duration_hours || 8;
      const daysNeeded = Math.ceil(duration / 8);

      const taskStartDate = new Date(startDate);
      taskStartDate.setDate(taskStartDate.getDate() + index);

      const taskEndDate = new Date(taskStartDate);
      taskEndDate.setDate(taskEndDate.getDate() + daysNeeded);

      return {
        ...task,
        startDate: taskStartDate.toISOString().split('T')[0],
        endDate: taskEndDate.toISOString().split('T')[0],
      };
    });

    // Valider que le planning contient les bonnes données
    if (!planning.orderedTasks || !Array.isArray(planning.orderedTasks)) {
      console.error('[AI Planning] Format de planning invalide');
      throw new Error('Format de planning invalide depuis OpenAI');
    }

    console.log('[AI Planning] Planning généré avec succès:', planning.orderedTasks.length, 'tâches');
    return planning;
  } catch (error) {
    // En cas d'erreur, utiliser la version basique mais avec un message d'erreur
    console.error('[AI Planning] Erreur OpenAI, fallback basique:', error);
    const basicPlanning = generateBasicPlanning(tasks, workers, deadline);
    basicPlanning.reasoning = `Erreur lors de l'appel OpenAI: ${error instanceof Error ? error.message : 'Erreur inconnue'}. Planning généré avec algorithme de base.`;
    return basicPlanning;
  }
}

/**
 * Version basique sans OpenAI (fallback)
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

