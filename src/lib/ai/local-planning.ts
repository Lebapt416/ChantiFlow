'use server';

import { calculateDaysNeeded, MAX_WORKING_HOURS_PER_DAY, LUNCH_BREAK_DURATION_HOURS } from './work-rules';

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
 * Génère un planning intelligent avec un algorithme local avancé
 * Analyse les dépendances, optimise l'ordre et assigne les ressources
 */
export async function generateLocalAIPlanning(
  tasks: Task[],
  workers: Worker[],
  deadline: string | null,
  siteName: string,
): Promise<PlanningResult> {
  if (tasks.length === 0) {
    return {
      orderedTasks: [],
      warnings: ['Aucune tâche à planifier'],
      reasoning: 'Aucune tâche disponible pour générer un planning.',
    };
  }

  // Phase 1 : Analyser les dépendances entre tâches
  const taskDependencies = analyzeDependencies(tasks);

  // Phase 2 : Classer les tâches par catégories logiques
  const categorizedTasks = categorizeTasks(tasks);

  // Phase 3 : Créer un graphe de dépendances
  const taskGraph = buildTaskGraph(tasks, taskDependencies);

  // Phase 4 : Ordonner les tâches selon le graphe (topological sort)
  const orderedTaskIds = topologicalSort(taskGraph);

  // Phase 5 : Calculer les dates en fonction de l'ordre et des durées
  const startDate = new Date();
  const deadlineDate = deadline ? new Date(deadline) : null;

  let currentDate = new Date(startDate);
  const taskSchedule: Array<{
    taskId: string;
    order: number;
    startDate: Date;
    endDate: Date;
    assignedWorkerId: string | null;
    priority: 'high' | 'medium' | 'low';
    dependencies: string[];
  }> = [];

  const warnings: string[] = [];

  orderedTaskIds.forEach((taskId, index) => {
    const task = tasks.find((t) => t.id === taskId);
    if (!task) return;

    const duration = task.duration_hours || 8;
    // Respecter la limite de 8h/jour avec pause déjeuner
    // Si une tâche dépasse 8h, elle doit être répartie sur plusieurs jours
    const daysNeeded = calculateDaysNeeded(duration, MAX_WORKING_HOURS_PER_DAY);
    
    // Avertir si une tâche dépasse 8h
    if (duration > MAX_WORKING_HOURS_PER_DAY) {
      warnings.push(
        `⚠️ La tâche "${task.title}" (${duration}h) sera répartie sur ${daysNeeded} jour(s) pour respecter la limite de ${MAX_WORKING_HOURS_PER_DAY}h/jour avec pause déjeuner.`,
      );
    }

    // Calculer la date de début en fonction des dépendances
    const dependencies = taskDependencies[taskId] || [];
    if (dependencies.length > 0) {
      const maxDependencyEnd = dependencies
        .map((depId) => {
          const depTask = taskSchedule.find((t) => t.taskId === depId);
          return depTask ? depTask.endDate : currentDate;
        })
        .reduce((max, date) => (date > max ? date : max), currentDate);

      currentDate = new Date(maxDependencyEnd);
      currentDate.setDate(currentDate.getDate() + 1); // Commencer le jour suivant
    }

    const taskStartDate = new Date(currentDate);
    const taskEndDate = new Date(currentDate);
    taskEndDate.setDate(taskEndDate.getDate() + daysNeeded);

    // Assigner un worker approprié
    const requiredRole = task.required_role;
    const assignedWorker = requiredRole
      ? workers.find((w) => w.role?.toLowerCase() === requiredRole.toLowerCase())
      : workers[0] || null;

    // Déterminer la priorité
    let priority: 'high' | 'medium' | 'low' = 'medium';
    if (index === 0 || categorizedTasks.preparation.includes(taskId)) {
      priority = 'high';
    } else if (categorizedTasks.finishing.includes(taskId)) {
      priority = 'low';
    } else if (dependencies.length > 0) {
      priority = 'high';
    }

    taskSchedule.push({
      taskId: task.id,
      order: index + 1,
      startDate: taskStartDate,
      endDate: taskEndDate,
      assignedWorkerId: assignedWorker?.id || null,
      priority,
      dependencies,
    });

    // Avancer la date pour la prochaine tâche
    currentDate = new Date(taskEndDate);
  });

  // Vérifier si la deadline est réaliste
  const estimatedEndDate = taskSchedule.length > 0
    ? taskSchedule[taskSchedule.length - 1].endDate
    : startDate;

  if (deadlineDate && estimatedEndDate > deadlineDate) {
    const daysOver = Math.ceil(
      (estimatedEndDate.getTime() - deadlineDate.getTime()) / (1000 * 60 * 60 * 24),
    );
    warnings.push(
      `La deadline du ${deadlineDate.toLocaleDateString('fr-FR')} semble irréaliste. Estimation: ${estimatedEndDate.toLocaleDateString('fr-FR')} (+${daysOver} jours)`,
    );
  }

  // Générer le raisonnement
  const reasoning = generateReasoning(
    tasks,
    workers,
    taskSchedule,
    categorizedTasks,
    taskDependencies,
  );

  return {
    orderedTasks: taskSchedule.map((task) => ({
      taskId: task.taskId,
      order: task.order,
      startDate: task.startDate.toISOString().split('T')[0],
      endDate: task.endDate.toISOString().split('T')[0],
      assignedWorkerId: task.assignedWorkerId,
      dependencies: task.dependencies,
      priority: task.priority,
    })),
    newDeadline: estimatedEndDate.toISOString().split('T')[0],
    warnings,
    reasoning,
  };
}

/**
 * Analyse les dépendances entre tâches en fonction de leur nom et contexte
 */
function analyzeDependencies(tasks: Task[]): Record<string, string[]> {
  const dependencies: Record<string, string[]> = {};

  tasks.forEach((task) => {
    const taskLower = task.title.toLowerCase();
    const deps: string[] = [];

    // Règles de dépendances basées sur les mots-clés
    if (taskLower.includes('peinture') || taskLower.includes('finition')) {
      // La peinture dépend de la structure
      const structureTask = tasks.find(
        (t) =>
          t.id !== task.id &&
          (t.title.toLowerCase().includes('structure') ||
            t.title.toLowerCase().includes('mur') ||
            t.title.toLowerCase().includes('cloison')),
      );
      if (structureTask) deps.push(structureTask.id);
    }

    if (taskLower.includes('électricité') || taskLower.includes('plomberie')) {
      // L'électricité/plomberie dépend de la structure
      const structureTask = tasks.find(
        (t) =>
          t.id !== task.id &&
          (t.title.toLowerCase().includes('structure') ||
            t.title.toLowerCase().includes('mur')),
      );
      if (structureTask) deps.push(structureTask.id);
    }

    if (taskLower.includes('carrelage') || taskLower.includes('sol')) {
      // Le carrelage dépend de la dalle/fondation
      const foundationTask = tasks.find(
        (t) =>
          t.id !== task.id &&
          (t.title.toLowerCase().includes('dalle') ||
            t.title.toLowerCase().includes('fondation')),
      );
      if (foundationTask) deps.push(foundationTask.id);
    }

    if (deps.length > 0) {
      dependencies[task.id] = deps;
    }
  });

  return dependencies;
}

/**
 * Catégorise les tâches par type (préparation, exécution, finition)
 */
function categorizeTasks(tasks: Task[]): {
  preparation: string[];
  execution: string[];
  finishing: string[];
} {
  const preparation: string[] = [];
  const execution: string[] = [];
  const finishing: string[] = [];

  const prepKeywords = [
    'fondation',
    'terrassement',
    'dalle',
    'structure',
    'gros œuvre',
    'préparation',
  ];
  const finishKeywords = [
    'peinture',
    'finition',
    'nettoyage',
    'réception',
    'carrelage',
    'revêtement',
  ];

  tasks.forEach((task) => {
    const taskLower = task.title.toLowerCase();
    if (prepKeywords.some((keyword) => taskLower.includes(keyword))) {
      preparation.push(task.id);
    } else if (finishKeywords.some((keyword) => taskLower.includes(keyword))) {
      finishing.push(task.id);
    } else {
      execution.push(task.id);
    }
  });

  return { preparation, execution, finishing };
}

/**
 * Construit un graphe de dépendances pour les tâches
 */
function buildTaskGraph(
  tasks: Task[],
  dependencies: Record<string, string[]>,
): Map<string, string[]> {
  const graph = new Map<string, string[]>();

  tasks.forEach((task) => {
    graph.set(task.id, dependencies[task.id] || []);
  });

  return graph;
}

/**
 * Tri topologique pour ordonner les tâches selon leurs dépendances
 */
function topologicalSort(graph: Map<string, string[]>): string[] {
  const visited = new Set<string>();
  const tempMark = new Set<string>();
  const result: string[] = [];

  function visit(node: string) {
    if (tempMark.has(node)) {
      // Cycle détecté, on ignore
      return;
    }
    if (visited.has(node)) {
      return;
    }

    tempMark.add(node);
    const deps = graph.get(node) || [];
    deps.forEach((dep) => visit(dep));
    tempMark.delete(node);
    visited.add(node);
    result.push(node);
  }

  graph.forEach((_, node) => {
    if (!visited.has(node)) {
      visit(node);
    }
  });

  return result;
}

/**
 * Génère une explication du raisonnement de l'algorithme
 */
function generateReasoning(
  tasks: Task[],
  workers: Worker[],
  schedule: Array<{
    taskId: string;
    order: number;
    priority: 'high' | 'medium' | 'low';
    dependencies: string[];
    assignedWorkerId: string | null;
  }>,
  categorized: ReturnType<typeof categorizeTasks>,
  dependencies: Record<string, string[]>,
): string {
  const totalTasks = schedule.length;
  const highPriorityCount = schedule.filter((t) => t.priority === 'high').length;
  const tasksWithDeps = schedule.filter((t) => t.dependencies.length > 0).length;
  const assignedWorkers = schedule.filter((t) => t.assignedWorkerId).length;

  let reasoning = `J'ai analysé ${totalTasks} tâches et généré un planning optimisé.\n\n`;

  if (categorized.preparation.length > 0) {
    reasoning += `• ${categorized.preparation.length} tâche(s) de préparation identifiée(s) (priorité haute)\n`;
  }

  if (tasksWithDeps > 0) {
    reasoning += `• ${tasksWithDeps} dépendance(s) détectée(s) entre les tâches\n`;
  }

  if (assignedWorkers > 0) {
    reasoning += `• ${assignedWorkers} tâche(s) assignée(s) à des workers selon leurs rôles\n`;
  }

  if (categorized.finishing.length > 0) {
    reasoning += `• ${categorized.finishing.length} tâche(s) de finition placée(s) en fin de planning\n`;
  }

  reasoning +=
    '\nLes tâches sont ordonnées selon un tri topologique qui respecte les dépendances. Les dates sont calculées en fonction des durées estimées et des contraintes de séquencement.';

  return reasoning;
}

