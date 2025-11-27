'use client';

import { useState, useMemo, useCallback } from 'react';
import { ChevronLeft, ChevronRight, Clock, X, AlertCircle } from 'lucide-react';

// Constantes pour les limites de travail
const MAX_WORKING_HOURS_PER_DAY = 8; // Maximum 8h de travail effectif par jour
const LUNCH_BREAK_DURATION_HOURS = 1; // 1h de pause déjeuner obligatoire

type PlanningTask = {
  taskId: string;
  taskTitle: string;
  order: number;
  startDate: string;
  endDate: string;
  assignedWorkerId?: string | null; // Ancien format (compatibilité)
  assignedWorkerIds?: string[]; // Nouveau format (collaboration)
  priority: 'high' | 'medium' | 'low';
  hours?: number;
  estimatedHours?: number;
};

type Worker = {
  id: string;
  name: string;
  email: string;
  role: string | null;
};

type TaskDetail = {
  taskId: string;
  title: string;
  status: string;
  progress?: number;
  description?: string;
};

type Props = {
  planning: PlanningTask[];
  workers: Worker[];
  taskDetails?: Record<string, TaskDetail>;
  onUpdate?: (taskId: string, workerId: string, day: string, hours: number) => void;
};

export function InteractiveCalendar({
  planning,
  workers,
  taskDetails = {},
  onUpdate,
}: Props) {
  const [currentWeekStart, setCurrentWeekStart] = useState(() => {
    const today = new Date();
    const dayOfWeek = today.getDay();
    const diff = today.getDate() - dayOfWeek + (dayOfWeek === 0 ? -6 : 1);
    const monday = new Date(today.setDate(diff));
    monday.setHours(0, 0, 0, 0);
    return monday;
  });

  const [selectedWorker, setSelectedWorker] = useState<Worker | null>(null);
  const [draggedTask, setDraggedTask] = useState<PlanningTask | null>(null);
  const [dragOverCell, setDragOverCell] = useState<{ workerId: string; day: string } | null>(
    null,
  );

  const weekDays = useMemo(() => {
    const days = [];
    for (let i = 0; i < 7; i++) {
      const date = new Date(currentWeekStart);
      date.setDate(date.getDate() + i);
      days.push(date);
    }
    return days;
  }, [currentWeekStart]);

  const nextWeek = () => {
    const next = new Date(currentWeekStart);
    next.setDate(next.getDate() + 7);
    setCurrentWeekStart(next);
  };

  const prevWeek = () => {
    const prev = new Date(currentWeekStart);
    prev.setDate(prev.getDate() - 7);
    setCurrentWeekStart(prev);
  };

  const today = () => {
    const today = new Date();
    const dayOfWeek = today.getDay();
    const diff = today.getDate() - dayOfWeek + (dayOfWeek === 0 ? -6 : 1);
    const monday = new Date(today.setDate(diff));
    monday.setHours(0, 0, 0, 0);
    setCurrentWeekStart(monday);
  };

  const tasksByDayAndWorker = useMemo(() => {
    const grouped: Record<
      string,
      Record<string, Array<PlanningTask & { hours: number }>>
    > = {};

    // Fonction pour distribuer les heures sur plusieurs jours
    const distributeHours = (totalHours: number, startDate: Date, endDate: Date, currentDay: Date): number => {
      if (totalHours <= MAX_WORKING_HOURS_PER_DAY) {
        // Si la tâche fait 8h ou moins, tout est sur le premier jour
        if (currentDay.toDateString() === startDate.toDateString()) {
          return totalHours;
        }
        return 0;
      }

      // Calculer le nombre de jours entre startDate et endDate (inclus)
      const daysDiff = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)) + 1;
      
      // Calculer le jour actuel dans la séquence (0 = premier jour)
      const currentDayIndex = Math.floor((currentDay.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
      
      if (currentDayIndex < 0 || currentDayIndex >= daysDiff) {
        return 0;
      }

      // Distribuer les heures : 8h par jour sauf le dernier jour
      if (currentDayIndex < daysDiff - 1) {
        return MAX_WORKING_HOURS_PER_DAY;
      } else {
        // Dernier jour : reste des heures
        const remainingHours = totalHours - (MAX_WORKING_HOURS_PER_DAY * (daysDiff - 1));
        return Math.max(0, remainingHours);
      }
    };

    weekDays.forEach((day) => {
      const dayKey = day.toISOString().split('T')[0];
      grouped[dayKey] = {};

      planning.forEach((task) => {
        const taskStart = new Date(task.startDate);
        taskStart.setHours(0, 0, 0, 0);
        const taskEnd = new Date(task.endDate);
        taskEnd.setHours(0, 0, 0, 0);
        const dayDate = new Date(day);
        dayDate.setHours(0, 0, 0, 0);

        // Vérifier si le jour est dans la plage de la tâche (inclus)
        if (dayDate >= taskStart && dayDate <= taskEnd) {
          const totalHours = task.hours || task.estimatedHours || 8;
          // Distribuer les heures réellement sur chaque jour
          const hoursForDay = distributeHours(totalHours, taskStart, taskEnd, dayDate);
          
          if (hoursForDay > 0) {
            // Utiliser assignedWorkerIds si disponible, sinon assignedWorkerId
            const workerIds = task.assignedWorkerIds || (task.assignedWorkerId ? [task.assignedWorkerId] : []);
            if (workerIds.length === 0) {
              const workerId = 'unassigned';
              if (!grouped[dayKey][workerId]) {
                grouped[dayKey][workerId] = [];
              }
              grouped[dayKey][workerId].push({
                ...task,
                hours: hoursForDay,
              });
            } else {
              // Assigner la tâche à tous les workers assignés
              workerIds.forEach(workerId => {
                if (!grouped[dayKey][workerId]) {
                  grouped[dayKey][workerId] = [];
                }
                grouped[dayKey][workerId].push({
                  ...task,
                  hours: hoursForDay,
                });
              });
            }
          }
        }
      });
    });
    
    return grouped;
  }, [planning, weekDays]);
  
  // Fonction pour obtenir le workerId principal (pour compatibilité)
  const getMainWorkerId = (task: PlanningTask): string => {
    if (task.assignedWorkerIds && task.assignedWorkerIds.length > 0) {
      return task.assignedWorkerIds[0];
    }
    return task.assignedWorkerId || 'unassigned';
  };

  const allWorkers = useMemo(() => {
    const workerMap = new Map<string, Worker>();
    workers.forEach((w) => workerMap.set(w.id, w));
    planning.forEach((task) => {
      // Gérer les deux formats (ancien et nouveau)
      const workerIds = task.assignedWorkerIds || (task.assignedWorkerId ? [task.assignedWorkerId] : []);
      workerIds.forEach(workerId => {
        const worker = workers.find((w) => w.id === workerId);
        if (worker) workerMap.set(worker.id, worker);
      });
    });
    return Array.from(workerMap.values());
  }, [workers, planning]);

  const handleDragStart = (task: PlanningTask) => {
    setDraggedTask(task);
  };

  const handleDragOver = (e: React.DragEvent, workerId: string, day: string) => {
    e.preventDefault();
    setDragOverCell({ workerId, day });
  };

  const handleDrop = (e: React.DragEvent, workerId: string, day: string) => {
    e.preventDefault();
    if (draggedTask && onUpdate) {
      onUpdate(draggedTask.taskId, workerId, day, draggedTask.hours || 8);
    }
    setDraggedTask(null);
    setDragOverCell(null);
  };

  const handleWorkerClick = (worker: Worker) => {
    setSelectedWorker(worker);
  };

  const dayNames = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'];
  const monthNames = [
    'Janvier',
    'Février',
    'Mars',
    'Avril',
    'Mai',
    'Juin',
    'Juillet',
    'Août',
    'Septembre',
    'Octobre',
    'Novembre',
    'Décembre',
  ];

  const weekRange = `${weekDays[0].getDate()} ${
    weekDays[0].getMonth() !== weekDays[6].getMonth()
      ? `${monthNames[weekDays[0].getMonth()].slice(0, 3)} - ${weekDays[6].getDate()} ${monthNames[weekDays[6].getMonth()].slice(0, 3)}`
      : monthNames[weekDays[0].getMonth()].slice(0, 3)
  } ${weekDays[0].getFullYear()}`;

  // Tâches assignées à l'employé sélectionné
  const workerTasks = selectedWorker
    ? planning.filter((task) => task.assignedWorkerId === selectedWorker.id)
    : [];

  return (
    <>
      <div className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
        {/* Header avec navigation */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Planning hebdomadaire
            </h3>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">{weekRange}</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={prevWeek}
              className="rounded-lg border border-zinc-200 p-2 transition hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              type="button"
              onClick={today}
              className="rounded-lg border border-zinc-200 px-3 py-2 text-xs font-medium transition hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
            >
              Aujourd&apos;hui
            </button>
            <button
              type="button"
              onClick={nextWeek}
              className="rounded-lg border border-zinc-200 p-2 transition hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Calendrier */}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="sticky left-0 z-10 border-b border-r border-zinc-200 bg-zinc-50 px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-400">
                  Employé
                </th>
                {weekDays.map((day, index) => {
                  const isToday = day.toDateString() === new Date().toDateString();
                  return (
                    <th
                      key={day.toISOString()}
                      className={`min-w-[120px] border-b border-r border-zinc-200 bg-zinc-50 px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider last:border-r-0 dark:border-zinc-700 dark:bg-zinc-900 ${
                        isToday
                          ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-900/20 dark:text-emerald-400'
                          : 'text-zinc-600 dark:text-zinc-400'
                      }`}
                    >
                      <div>{dayNames[index]}</div>
                      <div className="mt-1 text-lg font-bold">{day.getDate()}</div>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {allWorkers.length > 0 ? (
                allWorkers.map((worker) => (
                  <tr
                    key={worker.id}
                    className="border-b border-zinc-200 dark:border-zinc-700"
                  >
                    <td className="sticky left-0 z-10 border-r border-zinc-200 bg-white px-4 py-3 text-sm font-medium text-zinc-900 dark:border-zinc-700 dark:bg-zinc-900 dark:text-white">
                      <button
                        type="button"
                        onClick={() => handleWorkerClick(worker)}
                        className="text-left hover:underline"
                      >
                        <div>{worker.name}</div>
                        <div className="text-xs text-zinc-500 dark:text-zinc-400">
                          {worker.role || 'Rôle non défini'}
                        </div>
                      </button>
                    </td>
                    {weekDays.map((day) => {
                      const dayKey = day.toISOString().split('T')[0];
                      const dayTasks =
                        tasksByDayAndWorker[dayKey]?.[worker.id] || [];
                      // Les heures sont déjà distribuées correctement (max 8h/jour)
                      const totalHours = dayTasks.reduce((sum, task) => sum + task.hours, 0);
                      const exceedsLimit = totalHours > MAX_WORKING_HOURS_PER_DAY;
                      const isToday = day.toDateString() === new Date().toDateString();
                      const isDragOver =
                        dragOverCell?.workerId === worker.id && dragOverCell?.day === dayKey;

                      return (
                        <td
                          key={day.toISOString()}
                          onDragOver={(e) => handleDragOver(e, worker.id, dayKey)}
                          onDrop={(e) => handleDrop(e, worker.id, dayKey)}
                          className={`min-w-[120px] border-r border-zinc-200 bg-white p-2 align-top last:border-r-0 dark:border-zinc-700 dark:bg-zinc-900 ${
                            isToday ? 'bg-emerald-50/50 dark:bg-emerald-900/10' : ''
                          } ${isDragOver ? 'bg-blue-100 dark:bg-blue-900/30' : ''} ${
                            exceedsLimit ? 'bg-rose-50/50 dark:bg-rose-900/10' : ''
                          }`}
                        >
                          {dayTasks.length > 0 ? (
                            <div className="space-y-1">
                              {dayTasks.map((task) => {
                                // Les heures sont déjà distribuées (max 8h/jour par tâche)
                                return (
                                  <div
                                    key={task.taskId}
                                    draggable={!!onUpdate}
                                    onDragStart={() => handleDragStart(task)}
                                    className={`cursor-move rounded-lg border px-2 py-1.5 text-xs ${
                                      task.priority === 'high'
                                        ? 'border-rose-200 bg-rose-50 text-rose-900 dark:border-rose-900/60 dark:bg-rose-900/20 dark:text-rose-200'
                                        : task.priority === 'medium'
                                          ? 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900/60 dark:bg-amber-900/20 dark:text-amber-200'
                                          : 'border-zinc-200 bg-zinc-50 text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300'
                                    }`}
                                  >
                                    <div className="font-semibold">{task.taskTitle}</div>
                                    <div className="mt-0.5 flex items-center gap-1 text-[10px] opacity-75">
                                      <Clock className="h-3 w-3" />
                                      {task.hours}h
                                    </div>
                                  </div>
                                );
                              })}
                              {totalHours > 0 && (
                                <div className={`mt-1 text-[10px] font-semibold ${
                                  exceedsLimit 
                                    ? 'text-rose-600 dark:text-rose-400' 
                                    : 'text-zinc-500 dark:text-zinc-400'
                                }`}>
                                  Total: {totalHours}h
                                  {exceedsLimit && (
                                    <span className="ml-1 flex items-center gap-0.5" title={`Attention: ${totalHours}h ce jour (limite: ${MAX_WORKING_HOURS_PER_DAY}h/jour). Cela peut être dû à plusieurs tâches.`}>
                                      <AlertCircle className="h-3 w-3" />
                                    </span>
                                  )}
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="text-center text-xs text-zinc-400 dark:text-zinc-500">
                              —
                            </div>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))
              ) : (
                <tr>
                  <td
                    colSpan={8}
                    className="border-r border-zinc-200 bg-white px-4 py-8 text-center text-sm text-zinc-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-400"
                  >
                    Aucun employé assigné. Ajoutez des membres à l&apos;équipe pour voir le planning.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal employé */}
      {selectedWorker && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-md rounded-2xl border border-zinc-200 bg-white p-6 shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
                {selectedWorker.name}
              </h3>
              <button
                type="button"
                onClick={() => setSelectedWorker(null)}
                className="rounded-lg p-1 transition hover:bg-zinc-100 dark:hover:bg-zinc-800"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
                  Informations
                </p>
                <p className="mt-1 text-sm text-zinc-900 dark:text-white">
                  {selectedWorker.email}
                </p>
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  {selectedWorker.role || 'Rôle non défini'}
                </p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
                  Tâches assignées ({workerTasks.length})
                </p>
                <div className="mt-2 space-y-2">
                  {workerTasks.length > 0 ? (
                    workerTasks.map((task) => {
                      const detail = taskDetails[task.taskId];
                      return (
                        <div
                          key={task.taskId}
                          className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-700"
                        >
                          <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                            {task.taskTitle}
                          </p>
                          <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                            {new Date(task.startDate).toLocaleDateString('fr-FR')} →{' '}
                            {new Date(task.endDate).toLocaleDateString('fr-FR')}
                          </p>
                          {detail && (
                            <div className="mt-2">
                              <div className="mb-1 flex items-center justify-between text-xs">
                                <span className="text-zinc-500 dark:text-zinc-400">Avancement</span>
                                <span className="font-semibold text-zinc-900 dark:text-white">
                                  {detail.progress || 0}%
                                </span>
                              </div>
                              <div className="h-1.5 rounded-full bg-zinc-200 dark:bg-zinc-800">
                                <div
                                  className="h-full rounded-full bg-emerald-500"
                                  style={{ width: `${detail.progress || 0}%` }}
                                />
                              </div>
                              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                                Statut: {detail.status}
                              </p>
                            </div>
                          )}
                        </div>
                      );
                    })
                  ) : (
                    <p className="text-sm text-zinc-500 dark:text-zinc-400">
                      Aucune tâche assignée
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

