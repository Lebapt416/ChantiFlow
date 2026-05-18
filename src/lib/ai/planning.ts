'use server';

import { getPrediction } from '@/lib/ai/prediction';
import {
  getWorkRule,
  getEffectiveWorkingHours,
  calculateDaysNeeded,
  MAX_WORKING_HOURS_PER_DAY,
  LUNCH_BREAK_DURATION_HOURS,
} from '@/lib/ai/work-rules';
import {
  formatDateISO,
  parseDateISO,
  prochaineJourOuvre,
  ajouterJoursOuvres,
  finDeTache,
} from '@/lib/ai/jours-ouvres';

// ─────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────

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

/** Réponse attendue de l'API météo ML */
type RecommandationMeteo = {
  favorable: boolean;
  reason?: string;
  recommendation?: string;
};

type ReponseMeteo = {
  recommendations?: RecommandationMeteo[];
  best_dates?: string[];
  warnings?: string[];
};

// ─────────────────────────────────────────────────────────────
// CONSTANTES
// ─────────────────────────────────────────────────────────────

// Utiliser la constante depuis work-rules pour la cohérence
const BASE_WORKING_HOURS = MAX_WORKING_HOURS_PER_DAY;

// ─────────────────────────────────────────────────────────────
// CALCUL DE COMPLEXITÉ (inchangé)
// ─────────────────────────────────────────────────────────────

/**
 * Calcule un score de complexité du chantier entre 1 et 10.
 * Basé sur la diversité des métiers et la durée moyenne des tâches.
 */
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
  const durationScore  = Math.min(avgDuration / 4, 10);

  return Math.max(1, Math.min(10, Number(((diversityScore + durationScore) / 2).toFixed(2))));
}

// ─────────────────────────────────────────────────────────────
// GÉNÉRATION DU PLANNING
// ─────────────────────────────────────────────────────────────

/**
 * Génère un planning intelligent pour un chantier.
 *
 * Nouveautés par rapport à la version précédente :
 *  - Les dates respectent les jours ouvrés français (week-ends + fériés exclus).
 *  - Si une date calculée tombe sur un jour non ouvré, elle est automatiquement
 *    poussée au prochain jour ouvré.
 *  - Le formatage ISO YYYY-MM-DD utilise les composantes locales (pas UTC) pour
 *    éviter le bug de décalage de fuseau horaire.
 *  - Toutes les fonctionnalités existantes sont préservées :
 *    complexité, realityFactor IA, optimisation météo.
 *
 * @param tasks    - Liste des tâches du chantier
 * @param workers  - Liste des ouvriers disponibles
 * @param deadline - Date limite du chantier (YYYY-MM-DD ou null)
 * @param location - Localisation pour l'optimisation météo (optionnel)
 * @param options  - Options de calcul des jours ouvrés
 */
export async function generatePlanning(
  tasks: Task[],
  workers: Worker[],
  deadline: string | null,
  location?: string,
  options: { includeSaturdays?: boolean } = {},
): Promise<PlanningResult> {
  const { includeSaturdays = false } = options;

  // ── Filtrer les tâches en attente ──────────────────────────
  const pendingTasks = tasks.filter((task) => task.status === 'pending');

  if (pendingTasks.length === 0) {
    return {
      orderedTasks: [],
      warnings: ['Aucune tâche en attente à planifier'],
    };
  }

  // ── Classifier les tâches par ordre logique ────────────────
  const classifiedTasks = classifyTasksByLogic(pendingTasks);

  // ── Date de départ : prochain jour ouvré à partir d'aujourd'hui ──
  // Correction du bug toISOString() : on utilise new Date() (local) + prochaineJourOuvre()
  const startDate = prochaineJourOuvre(new Date(), includeSaturdays);

  // ── Calcul théorique du nombre de jours nécessaires ──────
  const totalHours   = classifiedTasks.reduce((sum, task) => sum + (task.duration_hours || 8), 0);
  const workingHoursPerDay = BASE_WORKING_HOURS;
  const daysNeeded   = Math.ceil(totalHours / workingHoursPerDay);

  // Date de fin estimée en jours ouvrés réels (plus toISOString())
  const estimatedEndDate = ajouterJoursOuvres(startDate, daysNeeded, includeSaturdays);

  // ── Avertissements ────────────────────────────────────────
  const warnings: string[] = [];

  // Tâches qui dépassent la limite quotidienne
  classifiedTasks.forEach((task) => {
    const taskHours = task.duration_hours || 8;
    if (taskHours > MAX_WORKING_HOURS_PER_DAY) {
      const joursTache = calculateDaysNeeded(taskHours, MAX_WORKING_HOURS_PER_DAY);
      warnings.push(
        `⚠️ La tâche "${task.title}" (${taskHours}h) sera répartie sur ${joursTache} jour(s) `
        + `pour respecter la limite de ${MAX_WORKING_HOURS_PER_DAY}h/jour `
        + `avec pause déjeuner de ${LUNCH_BREAK_DURATION_HOURS}h.`,
      );
    }
  });

  // Vérification deadline
  const deadlineDate = deadline ? parseDateISO(deadline) : null;
  if (deadlineDate && estimatedEndDate > deadlineDate) {
    warnings.push(
      `La deadline du ${deadlineDate.toLocaleDateString('fr-FR')} semble irréaliste. `
      + `Estimation : ${estimatedEndDate.toLocaleDateString('fr-FR')}`,
    );
  }

  // ── Facteur de réalité via prédiction IA (inchangé) ──────
  const complexity = computeComplexity(classifiedTasks);
  let realityFactor = 1;
  try {
    const predictionDays = await getPrediction(classifiedTasks.length, complexity);
    if (predictionDays > 0) {
      const theoreticalDays = Math.max(daysNeeded, 1);
      realityFactor = Math.max(1, Number((predictionDays / theoreticalDays).toFixed(2)));

      if (predictionDays > theoreticalDays * 1.1) {
        warnings.push(
          `⚠️ L'IA prévoit ${predictionDays} jours (basé sur l'historique). `
          + `Ajustement du planning appliqué.`,
        );
      }
    }
  } catch (error) {
    console.warn('Impossible de récupérer la prédiction IA :', error);
  }

  const adjustedDailyHours = BASE_WORKING_HOURS / realityFactor;

  // ─────────────────────────────────────────────────────────
  // GÉNÉRATION DES TÂCHES PLANIFIÉES (jours ouvrés)
  // ─────────────────────────────────────────────────────────
  //
  // Algorithme séquentiel avec curseur `currentDate` :
  //   - Chaque tâche démarre le jour ouvré courant.
  //   - Sa fin est calculée en jours ouvrés (finDeTache).
  //   - La tâche suivante démarre le jour ouvré après la fin.
  //
  // Avantage : pas besoin de recalculer depuis le début à chaque itération,
  // et le curseur franchit automatiquement les week-ends et fériés.

  /** Curseur de date courante (avance tâche par tâche) */
  let currentDate = new Date(startDate);

  const orderedTasks = classifiedTasks.map((task, index) => {
    // Sécurité : snapper au prochain jour ouvré (cas où currentDate aurait été
    // poussée sur un week-end ou un férié par l'optimisation météo)
    currentDate = prochaineJourOuvre(currentDate, includeSaturdays);
    const taskStart = new Date(currentDate);

    // Règles métier pour ce métier
    const workRule      = getWorkRule(task.required_role);
    const effectiveHours = getEffectiveWorkingHours(workRule);
    const taskHours     = task.duration_hours || effectiveHours;

    // Nombre de jours ouvrés nécessaires (plafond 8h/jour)
    const taskDays = calculateDaysNeeded(taskHours, MAX_WORKING_HOURS_PER_DAY);

    // Date de fin : le premier jour (taskStart) compte comme J1
    const taskEnd = finDeTache(taskStart, taskDays, includeSaturdays);

    // Faire avancer le curseur : la tâche suivante commence le jour ouvré d'après
    currentDate = ajouterJoursOuvres(taskEnd, 1, includeSaturdays);

    // ── Assignation du worker ────────────────────────────────
    const requiredRole   = task.required_role;
    const assignedWorker = requiredRole
      ? workers.find((w) => w.role?.toLowerCase() === requiredRole.toLowerCase())
      : workers[0] ?? null;

    // ── Priorité ─────────────────────────────────────────────
    let priority: 'high' | 'medium' | 'low' = 'medium';
    if (index === 0) priority = 'high';
    if (index >= classifiedTasks.length - 2) priority = 'low';

    return {
      taskId           : task.id,
      order            : index + 1,
      // formatDateISO() : composantes locales → pas de bug UTC
      startDate        : formatDateISO(taskStart),
      endDate          : formatDateISO(taskEnd),
      assignedWorkerId : assignedWorker?.id ?? null,
      dependencies     : [] as string[],
      priority,
    };
  });

  // ─────────────────────────────────────────────────────────
  // OPTIMISATION MÉTÉO (logique inchangée, types corrigés)
  // ─────────────────────────────────────────────────────────

  if (location && location.trim()) {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_PREDICTION_API_URL || process.env.ML_API_URL || '';
      if (apiUrl) {
        const { weatherCache } = await import('@/lib/ai/weather-cache');
        const locationKey = location.trim();
        console.log('🌤️ Optimisation météo pour :', locationKey);

        let weatherData: ReponseMeteo | null = null;

        const cachedData = weatherCache.get(locationKey) as ReponseMeteo | undefined;
        if (cachedData) {
          console.log('✅ Données météo récupérées depuis le cache');
          weatherData = cachedData;
        } else {
          console.log('🌐 Appel API météo (cache vide ou expiré)');
          warnings.push(`🌤️ Optimisation météo activée pour ${locationKey}`);

          const weatherOptimization = await fetch(
            `${apiUrl.replace(/\/$/, '')}/planning/optimize-weather`,
            {
              method : 'POST',
              headers: { 'Content-Type': 'application/json' },
              body   : JSON.stringify({
                tasks: orderedTasks.map((ot) => {
                  const task = classifiedTasks.find((t) => t.id === ot.taskId);
                  return {
                    task_role : task?.required_role ?? null,
                    task_title: task?.title ?? '',
                    planned_date: ot.startDate,
                  };
                }),
                location  : locationKey,
                start_date: formatDateISO(startDate),
              }),
              cache: 'no-store',
            },
          );

          if (weatherOptimization.ok) {
            weatherData = (await weatherOptimization.json()) as ReponseMeteo;
            console.log("✅ Données météo reçues depuis l'API");
            weatherCache.set(locationKey, weatherData);
          } else {
            const errorText = await weatherOptimization.text();
            console.warn('⚠️ Erreur API météo :', weatherOptimization.status, errorText);
            warnings.push('⚠️ Impossible de récupérer les prévisions météo pour optimiser le planning.');
          }
        }

        // ── Appliquer les recommandations météo ────────────────
        if (weatherData) {
          if (weatherData.recommendations && Array.isArray(weatherData.recommendations)) {
            weatherData.recommendations.forEach(
              (rec: RecommandationMeteo, idx: number) => {
                if (!rec.favorable) {
                  const currentTask = orderedTasks[idx];
                  const task        = classifiedTasks[idx];

                  if (currentTask && task) {
                    if (weatherData!.best_dates && weatherData!.best_dates.length > idx) {
                      // Analyser la meilleure date suggérée par la météo
                      // et la snapper au prochain jour ouvré
                      const bestDateBrute  = parseDateISO(weatherData!.best_dates[idx]);
                      const bestDateOuvre  = prochaineJourOuvre(bestDateBrute, includeSaturdays);
                      const daysDiff = Math.ceil(
                        (bestDateOuvre.getTime() - parseDateISO(currentTask.startDate).getTime())
                        / (1000 * 60 * 60 * 24),
                      );

                      if (daysDiff !== 0) {
                        // Recalculer start et end en jours ouvrés
                        const taskDaysMeteo = Math.ceil(
                          (task.duration_hours || 8) / adjustedDailyHours,
                        );
                        currentTask.startDate = formatDateISO(bestDateOuvre);
                        currentTask.endDate   = formatDateISO(
                          finDeTache(bestDateOuvre, taskDaysMeteo, includeSaturdays),
                        );

                        warnings.push(
                          `🌤️ ${rec.recommendation ?? `Tâche "${task.title}" décalée de ${Math.abs(daysDiff)} jour(s) pour conditions météo optimales (${rec.reason ?? 'pluie prévue'})`}`,
                        );
                      } else {
                        warnings.push(
                          `🌤️ Attention : ${rec.reason ?? 'Conditions météo défavorables'} `
                          + `pour "${task.title}" le ${currentTask.startDate}`,
                        );
                      }
                    } else {
                      warnings.push(
                        `🌤️ Conditions météo défavorables pour "${task.title}" `
                        + `le ${currentTask.startDate} : ${rec.reason ?? 'pluie prévue'}`,
                      );
                    }
                  }
                }
              },
            );
          }

          // Ajouter les avertissements de l'API météo
          if (weatherData.warnings && Array.isArray(weatherData.warnings)) {
            warnings.push(...weatherData.warnings);
          }
        }
      } else {
        console.warn("⚠️ URL API non configurée pour la météo");
      }
    } catch (error) {
      console.error('❌ Erreur optimisation météo :', error);
      warnings.push("⚠️ Erreur lors de l'optimisation météo. Planning généré sans optimisation.");
    }
  } else {
    console.log('ℹ️ Pas de localisation fournie, optimisation météo ignorée');
  }

  // ── Date de fin du planning global ───────────────────────
  const derniereEndDate = orderedTasks[orderedTasks.length - 1].endDate;

  return {
    orderedTasks,
    newDeadline: derniereEndDate,
    warnings,
  };
}

// ─────────────────────────────────────────────────────────────
// CLASSIFICATION DES TÂCHES (inchangée)
// ─────────────────────────────────────────────────────────────

/**
 * Classe les tâches par ordre logique BTP.
 * Identifie les dépendances implicites et optimise l'ordre d'exécution.
 * Cette fonction est purement synchrone et ne modifie pas l'original.
 */
function classifyTasksByLogic(tasks: Task[]): Task[] {
  const sortedTasks = [...tasks];

  sortedTasks.sort((a, b) => {
    // 1. Priorité aux tâches avec durée définie
    if (a.duration_hours && !b.duration_hours) return -1;
    if (!a.duration_hours && b.duration_hours) return 1;

    // 2. Tâches de préparation / fondations en premier
    const prepKeywords = ['fondation', 'structure', 'terrassement', 'préparation'];
    const aIsPrep = prepKeywords.some((kw) => a.title.toLowerCase().includes(kw));
    const bIsPrep = prepKeywords.some((kw) => b.title.toLowerCase().includes(kw));
    if (aIsPrep && !bIsPrep) return -1;
    if (!aIsPrep && bIsPrep) return 1;

    // 3. Tâches de finition en dernier
    const finishKeywords = ['peinture', 'finition', 'nettoyage', 'réception'];
    const aIsFinish = finishKeywords.some((kw) => a.title.toLowerCase().includes(kw));
    const bIsFinish = finishKeywords.some((kw) => b.title.toLowerCase().includes(kw));
    if (aIsFinish && !bIsFinish) return 1;
    if (!aIsFinish && bIsFinish) return -1;

    // 4. Tâches avec rôle spécifique avant les tâches générales
    if (a.required_role && !b.required_role) return -1;
    if (!a.required_role && b.required_role) return 1;

    // 5. Tâches plus longues en premier (meilleure répartition)
    if (a.duration_hours && b.duration_hours) {
      return b.duration_hours - a.duration_hours;
    }

    // 6. Ordre alphabétique comme dernier critère de stabilité
    return a.title.localeCompare(b.title, 'fr');
  });

  return sortedTasks;
}
