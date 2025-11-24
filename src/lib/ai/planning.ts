'use server';

import { getPrediction } from '@/lib/ai/prediction';
import { getWorkRule, getEffectiveWorkingHours } from '@/lib/ai/work-rules';

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

const BASE_WORKING_HOURS = 8;

function computeComplexity(tasks: Task[]): number {
  if (tasks.length === 0) return 1;

  const uniqueRoles = new Set(
    tasks
      .map((task) => task.required_role?.toLowerCase().trim())
      .filter((role): role is string => Boolean(role)),
  ).size;

  const totalDuration = tasks.reduce((sum, task) => sum + (task.duration_hours || 8), 0);
  const avgDuration = totalDuration / tasks.length;

  const diversityScore = Math.min(uniqueRoles + tasks.length / 5, 10);
  const durationScore = Math.min(avgDuration / 4, 10);

  return Math.max(1, Math.min(10, Number(((diversityScore + durationScore) / 2).toFixed(2))));
}

/**
 * Génère un planning intelligent pour un chantier
 * Classe les tâches par ordre logique et optimise l'utilisation des ressources
 */
export async function generatePlanning(
  tasks: Task[],
  workers: Worker[],
  deadline: string | null,
  location?: string, // Localisation du chantier pour la météo
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

  // Calculer la date de fin estimée (théorique)
  const workingHoursPerDay = BASE_WORKING_HOURS;
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

  const complexity = computeComplexity(classifiedTasks);
  let realityFactor = 1;
  try {
    const predictionDays = await getPrediction(classifiedTasks.length, complexity);
    if (predictionDays > 0) {
      const theoreticalDays = Math.max(daysNeeded, 1);
      realityFactor = Math.max(1, Number((predictionDays / theoreticalDays).toFixed(2)));

      if (predictionDays > theoreticalDays * 1.1) {
        warnings.push(
          `⚠️ L'IA prévoit ${predictionDays} jours (basé sur l'historique). Ajustement du planning appliqué.`,
        );
      }
    }
  } catch (error) {
    console.warn('Impossible de récupérer la prédiction IA:', error);
  }

  const adjustedDailyHours = BASE_WORKING_HOURS / realityFactor;

  // Générer le planning initial avec dates et assignations
  let orderedTasks = classifiedTasks.map((task, index) => {
    const taskStartDate = new Date(startDate);
    
    // Utiliser les heures effectives selon les règles de métier
    const workRule = getWorkRule(task.required_role);
    const effectiveHours = getEffectiveWorkingHours(workRule);
    const taskHours = task.duration_hours || effectiveHours;
    
    // Calculer la date de début en fonction de l'ordre et des dépendances
    const previousTasksHours = classifiedTasks
      .slice(0, index)
      .reduce((sum, t) => {
        const tRule = getWorkRule(t.required_role);
        const tEffectiveHours = getEffectiveWorkingHours(tRule);
        return sum + (t.duration_hours || tEffectiveHours);
      }, 0);
    
    taskStartDate.setDate(
      taskStartDate.getDate() + Math.floor(previousTasksHours / adjustedDailyHours),
    );

    const taskEndDate = new Date(taskStartDate);
    taskEndDate.setDate(
      taskEndDate.getDate() + Math.ceil(taskHours / adjustedDailyHours),
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

  // Optimiser avec la météo si la localisation est fournie
  if (location && location.trim()) {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_PREDICTION_API_URL || process.env.ML_API_URL || '';
      if (apiUrl) {
        console.log('🌤️ Optimisation météo pour:', location);
        warnings.push(`🌤️ Optimisation météo activée pour ${location}`);
        const weatherOptimization = await fetch(`${apiUrl.replace(/\/$/, '')}/planning/optimize-weather`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            tasks: orderedTasks.map((ot) => {
              const task = classifiedTasks.find((t) => t.id === ot.taskId);
              return {
                task_role: task?.required_role || null,
                task_title: task?.title || '',
                planned_date: ot.startDate,
              };
            }),
            location: location.trim(),
            start_date: startDate.toISOString().split('T')[0],
          }),
          cache: 'no-store',
        });

        if (weatherOptimization.ok) {
          const weatherData = await weatherOptimization.json();
          console.log('✅ Données météo reçues:', weatherData);
          
          // Appliquer les recommandations météo
          if (weatherData.recommendations && Array.isArray(weatherData.recommendations)) {
            weatherData.recommendations.forEach((rec: any, idx: number) => {
              if (!rec.favorable) {
                const currentTask = orderedTasks[idx];
                const task = classifiedTasks[idx];
                
                if (currentTask && task) {
                  // Chercher une meilleure date dans best_dates
                  if (weatherData.best_dates && weatherData.best_dates.length > idx) {
                    const bestDate = new Date(weatherData.best_dates[idx]);
                    const daysDiff = Math.ceil(
                      (bestDate.getTime() - new Date(currentTask.startDate).getTime()) /
                        (1000 * 60 * 60 * 24),
                    );
                    
                    if (daysDiff !== 0) {
                      currentTask.startDate = bestDate.toISOString().split('T')[0];
                      const endDate = new Date(bestDate);
                      endDate.setDate(endDate.getDate() + Math.ceil((task.duration_hours || 8) / adjustedDailyHours));
                      currentTask.endDate = endDate.toISOString().split('T')[0];
                      
                      warnings.push(
                        `🌤️ ${rec.recommendation || `Tâche "${task.title}" décalée de ${Math.abs(daysDiff)} jour(s) pour conditions météo optimales (${rec.reason || 'pluie prévue'})`}`,
                      );
                    } else {
                      warnings.push(
                        `🌤️ Attention: ${rec.reason || 'Conditions météo défavorables'} pour "${task.title}" le ${currentTask.startDate}`,
                      );
                    }
                  } else {
                    warnings.push(
                      `🌤️ Conditions météo défavorables pour "${task.title}" le ${currentTask.startDate}: ${rec.reason || 'pluie prévue'}`,
                    );
                  }
                }
              }
            });
          }

          // Ajouter les warnings de l'API
          if (weatherData.warnings && Array.isArray(weatherData.warnings)) {
            warnings.push(...weatherData.warnings);
          }
        } else {
          const errorText = await weatherOptimization.text();
          console.warn('⚠️ Erreur API météo:', weatherOptimization.status, errorText);
          warnings.push('⚠️ Impossible de récupérer les prévisions météo pour optimiser le planning.');
        }
      } else {
        console.warn('⚠️ URL API non configurée pour la météo');
      }
    } catch (error) {
      console.error('❌ Erreur optimisation météo:', error);
      warnings.push('⚠️ Erreur lors de l\'optimisation météo. Planning généré sans optimisation.');
    }
  } else {
    console.log('ℹ️ Pas de localisation fournie, optimisation météo ignorée');
  }

  const lastTaskEnd = new Date(orderedTasks[orderedTasks.length - 1].endDate);

  return {
    orderedTasks,
    newDeadline: lastTaskEnd.toISOString().split('T')[0],
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

