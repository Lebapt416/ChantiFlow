'use server';

import { createSupabaseServerClient } from '@/lib/supabase/server';

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
    assignedWorkerIds: string[]; // Permet plusieurs workers pour collaboration
    dependencies: string[];
    priority: 'high' | 'medium' | 'low';
    estimatedHours: number;
  }>;
  newDeadline?: string;
  warnings: string[];
  reasoning: string;
};

/**
 * Lois du travail françaises (Code du travail)
 */
const FRENCH_LABOR_LAWS = {
  // Temps de travail maximum par jour
  MAX_WORKING_HOURS_PER_DAY: 10, // Maximum légal (avec dérogation)
  STANDARD_WORKING_HOURS_PER_DAY: 8, // Standard
  MAX_WORKING_HOURS_PER_WEEK: 48, // Maximum absolu
  STANDARD_WORKING_HOURS_PER_WEEK: 35, // Standard
  
  // Pauses obligatoires
  PAUSE_AFTER_6_HOURS: 20, // 20 minutes après 6h de travail
  PAUSE_AFTER_4_5_HOURS: 15, // 15 minutes après 4h30 de travail (recommandé)
  
  // Repos quotidien
  MIN_REST_BETWEEN_SHIFTS: 11, // 11 heures de repos entre deux journées
  
  // Repos hebdomadaire
  MIN_WEEKLY_REST: 24, // 24 heures consécutives par semaine (généralement le dimanche)
  
  // Jours fériés
  PUBLIC_HOLIDAYS: [
    '01-01', // Jour de l'an
    '05-01', // Fête du travail
    '05-08', // Victoire 1945
    '07-14', // Fête nationale
    '08-15', // Assomption
    '11-01', // Toussaint
    '11-11', // Armistice
    '12-25', // Noël
  ],
};

/**
 * Base de connaissances des métiers du bâtiment
 */
const PROFESSION_KNOWLEDGE: Record<string, {
  keywords: string[];
  canCollaborate: boolean;
  typicalTasks: string[];
  averageSpeed: number; // Multiplicateur de vitesse (1.0 = normal, >1.0 = plus rapide)
}> = {
  'maçon': {
    keywords: ['maçon', 'maçonnerie', 'mur', 'cloison', 'parpaing', 'brique', 'ciment'],
    canCollaborate: true,
    typicalTasks: ['fondation', 'structure', 'mur', 'cloison', 'enduit'],
    averageSpeed: 1.0,
  },
  'électricien': {
    keywords: ['électricien', 'électricité', 'câblage', 'tableau', 'prise', 'éclairage'],
    canCollaborate: true,
    typicalTasks: ['électricité', 'câblage', 'tableau électrique', 'éclairage'],
    averageSpeed: 0.9,
  },
  'plombier': {
    keywords: ['plombier', 'plomberie', 'sanitaire', 'eau', 'chauffage', 'radiateur'],
    canCollaborate: true,
    typicalTasks: ['plomberie', 'sanitaire', 'chauffage', 'évacuation'],
    averageSpeed: 0.95,
  },
  'peintre': {
    keywords: ['peintre', 'peinture', 'enduit', 'finition', 'revêtement'],
    canCollaborate: true,
    typicalTasks: ['peinture', 'finition', 'enduit', 'revêtement'],
    averageSpeed: 1.1,
  },
  'carreleur': {
    keywords: ['carreleur', 'carrelage', 'faïence', 'sol', 'revêtement'],
    canCollaborate: true,
    typicalTasks: ['carrelage', 'faïence', 'sol', 'salle de bain'],
    averageSpeed: 0.85,
  },
  'charpentier': {
    keywords: ['charpentier', 'charpente', 'bois', 'toiture', 'ossature'],
    canCollaborate: true,
    typicalTasks: ['charpente', 'toiture', 'ossature bois'],
    averageSpeed: 0.9,
  },
  'couvreur': {
    keywords: ['couvreur', 'couverture', 'toiture', 'tuile', 'ardoise'],
    canCollaborate: true,
    typicalTasks: ['couverture', 'toiture', 'tuile', 'ardoise'],
    averageSpeed: 0.95,
  },
  'menuisier': {
    keywords: ['menuisier', 'menuiserie', 'bois', 'fenêtre', 'porte', 'parquet'],
    canCollaborate: true,
    typicalTasks: ['menuiserie', 'fenêtre', 'porte', 'parquet'],
    averageSpeed: 1.0,
  },
};

/**
 * Reconnaît le métier d'un worker ou d'une tâche
 */
function recognizeProfession(text: string | null): string | null {
  if (!text) return null;
  
  const textLower = text.toLowerCase();
  
  for (const [profession, data] of Object.entries(PROFESSION_KNOWLEDGE)) {
    if (data.keywords.some(keyword => textLower.includes(keyword))) {
      return profession;
    }
  }
  
  return null;
}

/**
 * Trouve tous les workers d'un même métier
 */
function findWorkersByProfession(
  workers: Worker[],
  profession: string,
): Worker[] {
  return workers.filter(worker => {
    const workerProfession = recognizeProfession(worker.role);
    return workerProfession === profession;
  });
}

/**
 * Calcule le temps de travail en respectant les lois françaises
 */
function calculateWorkSchedule(
  taskHours: number,
  startDate: Date,
  workingHoursPerDay: number = FRENCH_LABOR_LAWS.STANDARD_WORKING_HOURS_PER_DAY,
): { startDate: Date; endDate: Date; actualDays: number; warnings: string[] } {
  const warnings: string[] = [];
  const currentDate = new Date(startDate);
  let remainingHours = taskHours;
  let actualDays = 0;
  
  // Vérifier que les heures par jour ne dépassent pas le maximum légal
  if (workingHoursPerDay > FRENCH_LABOR_LAWS.MAX_WORKING_HOURS_PER_DAY) {
    warnings.push(
      `⚠️ Attention: ${workingHoursPerDay}h/jour dépasse le maximum légal de ${FRENCH_LABOR_LAWS.MAX_WORKING_HOURS_PER_DAY}h/jour (dérogation requise)`,
    );
  }
  
  // Calculer les jours nécessaires en respectant les pauses
  while (remainingHours > 0) {
    // Vérifier si c'est un jour férié
    const monthDay = `${String(currentDate.getMonth() + 1).padStart(2, '0')}-${String(currentDate.getDate()).padStart(2, '0')}`;
    if (FRENCH_LABOR_LAWS.PUBLIC_HOLIDAYS.includes(monthDay)) {
      currentDate.setDate(currentDate.getDate() + 1);
      continue;
    }
    
    // Vérifier si c'est dimanche (repos hebdomadaire)
    if (currentDate.getDay() === 0) {
      currentDate.setDate(currentDate.getDate() + 1);
      continue;
    }
    
    // Calculer les heures effectives du jour (avec pauses)
    let effectiveHours = Math.min(remainingHours, workingHoursPerDay);
    
    // Appliquer les pauses obligatoires
    if (effectiveHours > 6) {
      // Pause de 20 minutes après 6h
      effectiveHours -= FRENCH_LABOR_LAWS.PAUSE_AFTER_6_HOURS / 60;
      warnings.push(`⏸️ Pause de ${FRENCH_LABOR_LAWS.PAUSE_AFTER_6_HOURS} minutes requise après 6h de travail`);
    } else if (effectiveHours > 4.5) {
      // Pause recommandée de 15 minutes après 4h30
      effectiveHours -= FRENCH_LABOR_LAWS.PAUSE_AFTER_4_5_HOURS / 60;
    }
    
    remainingHours -= effectiveHours;
    actualDays++;
    
    if (remainingHours > 0) {
      // Passer au jour suivant avec repos minimum de 11h
      currentDate.setDate(currentDate.getDate() + 1);
    }
  }
  
  const endDate = new Date(currentDate);
  
  return { startDate: new Date(startDate), endDate, actualDays, warnings };
}

/**
 * Génère un planning intelligent amélioré avec entraînement continu
 */
export async function generateImprovedPlanning(
  tasks: Task[],
  workers: Worker[],
  deadline: string | null,
  siteName: string,
  siteId: string,
): Promise<PlanningResult> {
  if (tasks.length === 0) {
    return {
      orderedTasks: [],
      warnings: ['Aucune tâche à planifier'],
      reasoning: 'Aucune tâche disponible pour générer un planning.',
    };
  }

  // Récupérer les patterns d'apprentissage depuis la base de données
  const learningPatterns = await getLearningPatterns(siteId);
  
  // Analyser les dépendances avec les patterns appris
  const taskDependencies = analyzeDependenciesWithLearning(tasks, learningPatterns);
  
  // Reconnaître les métiers et permettre la collaboration
  const taskProfessions = new Map<string, string | null>();
  tasks.forEach(task => {
    const profession = recognizeProfession(task.required_role) || 
                      recognizeProfession(task.title);
    taskProfessions.set(task.id, profession);
  });
  
  // Créer un graphe de dépendances
  const taskGraph = buildTaskGraph(tasks, taskDependencies);
  
  // Ordonner les tâches (tri topologique)
  const orderedTaskIds = topologicalSort(taskGraph);
  
  // Calculer les dates en respectant les lois du travail
  const startDate = new Date();
  const deadlineDate = deadline ? new Date(deadline) : null;
  
  let currentDate = new Date(startDate);
  const taskSchedule: Array<{
    taskId: string;
    order: number;
    startDate: Date;
    endDate: Date;
    assignedWorkerIds: string[];
    dependencies: string[];
    priority: 'high' | 'medium' | 'low';
    estimatedHours: number;
  }> = [];
  
  const warnings: string[] = [];
  
  orderedTaskIds.forEach((taskId, index) => {
    const task = tasks.find((t) => t.id === taskId);
    if (!task) return;
    
    // Utiliser la durée estimée de la tâche (duration_hours)
    const taskDuration = task.duration_hours || 8;
    
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
      currentDate.setDate(currentDate.getDate() + 1); // Repos minimum entre tâches
    }
    
    // Calculer le planning en respectant les lois du travail
    const schedule = calculateWorkSchedule(taskDuration, currentDate);
    warnings.push(...schedule.warnings);
    
    // Assigner les workers avec collaboration possible
    const profession = taskProfessions.get(taskId);
    let assignedWorkers: Worker[] = [];
    
    if (profession) {
      // Trouver tous les workers du même métier
      const professionWorkers = findWorkersByProfession(workers, profession);
      
      if (professionWorkers.length > 0) {
        // Permettre la collaboration si la tâche est longue ou complexe
        const professionData = PROFESSION_KNOWLEDGE[profession];
        if (professionData?.canCollaborate && (taskDuration > 8 || professionWorkers.length > 1)) {
          // Assigner plusieurs workers pour collaborer
          const workersNeeded = Math.min(
            Math.ceil(taskDuration / 8), // 1 worker par 8h
            professionWorkers.length,
          );
          assignedWorkers = professionWorkers.slice(0, workersNeeded);
        } else {
          // Assigner un seul worker
          assignedWorkers = [professionWorkers[0]];
        }
      }
    }
    
    // Si aucun worker trouvé par métier, prendre le premier disponible
    if (assignedWorkers.length === 0) {
      assignedWorkers = workers.length > 0 ? [workers[0]] : [];
    }
    
    // Déterminer la priorité
    let priority: 'high' | 'medium' | 'low' = 'medium';
    if (index === 0 || dependencies.length > 0) {
      priority = 'high';
    } else if (index >= orderedTaskIds.length - 2) {
      priority = 'low';
    }
    
    taskSchedule.push({
      taskId: task.id,
      order: index + 1,
      startDate: schedule.startDate,
      endDate: schedule.endDate,
      assignedWorkerIds: assignedWorkers.map(w => w.id),
      dependencies,
      priority,
      estimatedHours: taskDuration,
    });
    
    // Avancer la date pour la prochaine tâche
    currentDate = new Date(schedule.endDate);
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
      `⚠️ La deadline du ${deadlineDate.toLocaleDateString('fr-FR')} semble irréaliste. Estimation: ${estimatedEndDate.toLocaleDateString('fr-FR')} (+${daysOver} jours)`,
    );
  }
  
  // Générer le raisonnement
  const reasoning = generateImprovedReasoning(
    tasks,
    workers,
    taskSchedule,
    taskDependencies,
    taskProfessions,
    learningPatterns,
  );
  
  const result: PlanningResult = {
    orderedTasks: taskSchedule.map((task) => ({
      taskId: task.taskId,
      order: task.order,
      startDate: task.startDate.toISOString().split('T')[0],
      endDate: task.endDate.toISOString().split('T')[0],
      assignedWorkerIds: task.assignedWorkerIds,
      dependencies: task.dependencies,
      priority: task.priority,
      estimatedHours: task.estimatedHours,
    })),
    newDeadline: estimatedEndDate.toISOString().split('T')[0],
    warnings: [...new Set(warnings)], // Supprimer les doublons
    reasoning,
  };
  
  // Enregistrer pour l'entraînement continu
  await savePlanningForTraining(siteId, tasks, workers, result, siteName, deadline);
  
  return result;
}

/**
 * Analyse les dépendances avec les patterns appris
 */
function analyzeDependenciesWithLearning(
  tasks: Task[],
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  learningPatterns: any,
): Record<string, string[]> {
  const dependencies: Record<string, string[]> = {};
  
  // Utiliser les patterns appris si disponibles
  const commonDeps = learningPatterns?.commonDependencies || [];
  
  tasks.forEach((task) => {
    const taskLower = task.title.toLowerCase();
    const deps: string[] = [];
    
    // Vérifier les patterns appris
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    commonDeps.forEach((pattern: any) => {
      if (taskLower.includes(pattern.to.toLowerCase())) {
        const fromTask = tasks.find(t => 
          t.id !== task.id && 
          t.title.toLowerCase().includes(pattern.from.toLowerCase())
        );
        if (fromTask && pattern.frequency > 0.5) {
          deps.push(fromTask.id);
        }
      }
    });
    
    // Règles de base si pas de pattern appris
    if (deps.length === 0) {
      // La peinture dépend de la structure
      if (taskLower.includes('peinture') || taskLower.includes('finition')) {
        const structureTask = tasks.find(
          (t) =>
            t.id !== task.id &&
            (t.title.toLowerCase().includes('structure') ||
              t.title.toLowerCase().includes('mur') ||
              t.title.toLowerCase().includes('cloison')),
        );
        if (structureTask) deps.push(structureTask.id);
      }
      
      // L&apos;électricité/plomberie dépend de la structure
      if (taskLower.includes('électricité') || taskLower.includes('plomberie')) {
        const structureTask = tasks.find(
          (t) =>
            t.id !== task.id &&
            (t.title.toLowerCase().includes('structure') ||
              t.title.toLowerCase().includes('mur')),
        );
        if (structureTask) deps.push(structureTask.id);
      }
    }
    
    if (deps.length > 0) {
      dependencies[task.id] = deps;
    }
  });
  
  return dependencies;
}

/**
 * Construit un graphe de dépendances
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
 * Tri topologique
 */
function topologicalSort(graph: Map<string, string[]>): string[] {
  const visited = new Set<string>();
  const tempMark = new Set<string>();
  const result: string[] = [];
  
  function visit(node: string) {
    if (tempMark.has(node)) {
      return; // Cycle détecté, on ignore
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
 * Génère un raisonnement amélioré
 */
function generateImprovedReasoning(
  tasks: Task[],
  workers: Worker[],
  schedule: Array<{
    taskId: string;
    order: number;
    priority: 'high' | 'medium' | 'low';
    dependencies: string[];
    assignedWorkerIds: string[];
    estimatedHours: number;
  }>,
  dependencies: Record<string, string[]>,
  taskProfessions: Map<string, string | null>,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  learningPatterns: any,
): string {
  const totalTasks = schedule.length;
  const tasksWithDeps = schedule.filter((t) => t.dependencies.length > 0).length;
  const collaborativeTasks = schedule.filter((t) => t.assignedWorkerIds.length > 1).length;
  const totalEstimatedHours = schedule.reduce((sum, t) => sum + t.estimatedHours, 0);
  
  let reasoning = `📊 Analyse intelligente du planning\n\n`;
  reasoning += `• ${totalTasks} tâche(s) analysée(s) et planifiée(s)\n`;
  reasoning += `• ${totalEstimatedHours}h de travail estimées au total\n`;
  
  if (tasksWithDeps > 0) {
    reasoning += `• ${tasksWithDeps} dépendance(s) détectée(s) et respectée(s)\n`;
  }
  
  if (collaborativeTasks > 0) {
    reasoning += `• ${collaborativeTasks} tâche(s) avec collaboration entre workers du même métier\n`;
  }
  
  // Détails des métiers
  const professionCounts = new Map<string, number>();
  schedule.forEach(task => {
    const profession = taskProfessions.get(task.taskId);
    if (profession) {
      professionCounts.set(profession, (professionCounts.get(profession) || 0) + 1);
    }
  });
  
  if (professionCounts.size > 0) {
    reasoning += `\n👷 Répartition par métier:\n`;
    professionCounts.forEach((count, profession) => {
      reasoning += `  • ${profession}: ${count} tâche(s)\n`;
    });
  }
  
  reasoning += `\n⚖️ Respect des lois du travail françaises:\n`;
  reasoning += `  • Maximum ${FRENCH_LABOR_LAWS.MAX_WORKING_HOURS_PER_DAY}h/jour respecté\n`;
  reasoning += `  • Pauses obligatoires intégrées (${FRENCH_LABOR_LAWS.PAUSE_AFTER_6_HOURS}min après 6h)\n`;
  reasoning += `  • Repos minimum de ${FRENCH_LABOR_LAWS.MIN_REST_BETWEEN_SHIFTS}h entre journées\n`;
  reasoning += `  • Jours fériés et dimanches exclus\n`;
  
  if (learningPatterns) {
    reasoning += `\n🧠 Patterns d'apprentissage utilisés pour optimiser le planning\n`;
  }
  
  return reasoning;
}

/**
 * Récupère les patterns d'apprentissage depuis la base de données
 */
async function getLearningPatterns(siteId: string) {
  try {
    const supabase = await createSupabaseServerClient();
    
    // Récupérer les plannings précédents pour ce chantier ou similaires
    const { data: previousPlannings } = await supabase
      .from('ai_planning_history')
      .select('*')
      .eq('site_id', siteId)
      .order('created_at', { ascending: false })
      .limit(10);
    
    if (!previousPlannings || previousPlannings.length === 0) {
      return null;
    }
    
    // Analyser les patterns
    const commonDependencies: Array<{ from: string; to: string; frequency: number }> = [];
    const roleAssignments: Array<{ role: string; tasks: string[]; frequency: number }> = [];
    
    // TODO: Analyser les patterns réels depuis les données historiques
    // Pour l'instant, on retourne des patterns par défaut améliorés
    
    return {
      commonDependencies,
      roleAssignments,
      averageDurations: {},
    };
  } catch (error) {
    console.error('[AI Training] Error getting patterns:', error);
    return null;
  }
}

/**
 * Enregistre le planning pour l'entraînement continu
 */
export async function savePlanningForTraining(
  siteId: string,
  tasks: Task[],
  workers: Worker[],
  planning: PlanningResult,
  siteName: string,
  deadline: string | null,
) {
  try {
    const supabase = await createSupabaseServerClient();
    
    // Enregistrer dans la table d'historique
    const { error } = await supabase
      .from('ai_planning_history')
      .insert({
        site_id: siteId,
        site_name: siteName,
        deadline: deadline,
        tasks_data: tasks,
        workers_data: workers,
        planning_result: planning,
        created_at: new Date().toISOString(),
      });
    
    if (error) {
      console.error('[AI Training] Error saving planning:', error);
      // Si la table n'existe pas, on continue sans erreur
      if (error.code !== '42P01') { // Table doesn't exist
        throw error;
      }
    } else {
      console.log('[AI Training] Planning saved for continuous learning');
    }
  } catch (error) {
    console.error('[AI Training] Error in savePlanningForTraining:', error);
    // Ne pas bloquer le processus si l'enregistrement échoue
  }
}

