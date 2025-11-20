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
};

/**
 * Génère un planning intelligent pour un chantier
 * Classe les tâches par ordre logique et optimise l'utilisation des ressources
 */
export async function generatePlanning(
  tasks: Task[],
  workers: Worker[],
  deadline: string | null,
): Promise<PlanningResult> {
  // Filtrer uniquement les tâches en attente
  const pendingTasks = tasks.filter((task) => task.status === 'pending');

  if (pendingTasks.length === 0) {
    return {
      orderedTasks: [],
      warnings: ['Aucune tâche en attente à planifier'],
    };
  }

  // Analyser les dépendances et classer par ordre logique
  const classifiedTasks = classifyTasksByLogic(pendingTasks);

  // Calculer les dates en fonction de la deadline
  const deadlineDate = deadline ? new Date(deadline) : null;
  const startDate = new Date();
  const totalHours = classifiedTasks.reduce(
    (sum, task) => sum + (task.duration_hours || 8),
    0,
  );

  // Calculer la date de fin estimée
  const workingHoursPerDay = 8;
  const daysNeeded = Math.ceil(totalHours / workingHoursPerDay);
  const estimatedEndDate = new Date(startDate);
  estimatedEndDate.setDate(estimatedEndDate.getDate() + daysNeeded);

  // Vérifier si la deadline est réaliste
  const warnings: string[] = [];
  if (deadlineDate && estimatedEndDate > deadlineDate) {
    warnings.push(
      `La deadline du ${deadlineDate.toLocaleDateString('fr-FR')} semble irréaliste. Estimation: ${estimatedEndDate.toLocaleDateString('fr-FR')}`,
    );
  }

  // Générer le planning avec dates et assignations
  const orderedTasks = classifiedTasks.map((task, index) => {
    const taskStartDate = new Date(startDate);
    // Calculer la date de début en fonction de l'ordre et des dépendances
    const previousTasksHours = classifiedTasks
      .slice(0, index)
      .reduce((sum, t) => sum + (t.duration_hours || 8), 0);
    taskStartDate.setDate(
      taskStartDate.getDate() + Math.floor(previousTasksHours / workingHoursPerDay),
    );

    const taskEndDate = new Date(taskStartDate);
    taskEndDate.setDate(
      taskEndDate.getDate() + Math.ceil((task.duration_hours || 8) / workingHoursPerDay),
    );

    // Trouver un worker approprié
    const requiredRole = task.required_role;
    const assignedWorker = requiredRole
      ? workers.find((w) => w.role?.toLowerCase() === requiredRole.toLowerCase())
      : workers[0] || null;

    // Déterminer la priorité
    let priority: 'high' | 'medium' | 'low' = 'medium';
    if (index === 0) priority = 'high';
    if (index >= classifiedTasks.length - 2) priority = 'low';

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
    newDeadline: estimatedEndDate.toISOString().split('T')[0],
    warnings,
  };
}

/**
 * Classe les tâches par ordre logique
 * Identifie les dépendances et optimise l'ordre d'exécution
 */
function classifyTasksByLogic(tasks: Task[]): Task[] {
  // Créer une copie pour ne pas modifier l'original
  const sortedTasks = [...tasks];

  // Règles de classement logique
  sortedTasks.sort((a, b) => {
    // 1. Priorité aux tâches avec durée définie
    if (a.duration_hours && !b.duration_hours) return -1;
    if (!a.duration_hours && b.duration_hours) return 1;

    // 2. Tâches de préparation en premier (fondations, structure)
    const prepKeywords = ['fondation', 'structure', 'terrassement', 'préparation'];
    const aIsPrep = prepKeywords.some((keyword) =>
      a.title.toLowerCase().includes(keyword),
    );
    const bIsPrep = prepKeywords.some((keyword) =>
      b.title.toLowerCase().includes(keyword),
    );
    if (aIsPrep && !bIsPrep) return -1;
    if (!aIsPrep && bIsPrep) return 1;

    // 3. Tâches de finition en dernier
    const finishKeywords = ['peinture', 'finition', 'nettoyage', 'réception'];
    const aIsFinish = finishKeywords.some((keyword) =>
      a.title.toLowerCase().includes(keyword),
    );
    const bIsFinish = finishKeywords.some((keyword) =>
      b.title.toLowerCase().includes(keyword),
    );
    if (aIsFinish && !bIsFinish) return 1;
    if (!aIsFinish && bIsFinish) return -1;

    // 4. Tâches avec rôle spécifique avant les tâches générales
    if (a.required_role && !b.required_role) return -1;
    if (!a.required_role && b.required_role) return 1;

    // 5. Tâches plus longues en premier (pour mieux répartir)
    if (a.duration_hours && b.duration_hours) {
      return b.duration_hours - a.duration_hours;
    }

    // 6. Ordre alphabétique comme dernier critère
    return a.title.localeCompare(b.title);
  });

  return sortedTasks;
}

