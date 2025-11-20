'use client';

import { useState, useMemo } from 'react';
import { ChevronLeft, ChevronRight, Clock } from 'lucide-react';

type PlanningTask = {
  taskId: string;
  taskTitle: string;
  order: number;
  startDate: string;
  endDate: string;
  assignedWorkerId: string | null;
  priority: 'high' | 'medium' | 'low';
};

type Worker = {
  id: string;
  name: string;
  email: string;
  role: string | null;
};

type Props = {
  planning: PlanningTask[];
  workers: Worker[];
};

export function WeeklyCalendar({ planning, workers }: Props) {
  const [currentWeekStart, setCurrentWeekStart] = useState(() => {
    const today = new Date();
    const dayOfWeek = today.getDay();
    const diff = today.getDate() - dayOfWeek + (dayOfWeek === 0 ? -6 : 1); // Lundi
    const monday = new Date(today.setDate(diff));
    monday.setHours(0, 0, 0, 0);
    return monday;
  });

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

  // Grouper les tâches par jour et worker
  const tasksByDayAndWorker = useMemo(() => {
    const grouped: Record<
      string,
      Record<string, Array<PlanningTask & { hours: number }>>
    > = {};

    weekDays.forEach((day) => {
      const dayKey = day.toISOString().split('T')[0];
      grouped[dayKey] = {};

      planning.forEach((task) => {
        const taskStart = new Date(task.startDate);
        const taskEnd = new Date(task.endDate);
        const dayDate = new Date(day);
        dayDate.setHours(0, 0, 0, 0);

        // Vérifier si la tâche chevauche ce jour
        if (dayDate >= taskStart && dayDate < taskEnd) {
          const workerId = task.assignedWorkerId || 'unassigned';
          if (!grouped[dayKey][workerId]) {
            grouped[dayKey][workerId] = [];
          }

          // Calculer les heures pour ce jour (simplifié : 8h par jour)
          const hours = 8;
          grouped[dayKey][workerId].push({ ...task, hours });
        }
      });
    });

    return grouped;
  }, [planning, weekDays]);

  const allWorkers = useMemo(() => {
    const workerMap = new Map<string, Worker>();
    workers.forEach((w) => workerMap.set(w.id, w));
    planning.forEach((task) => {
      if (task.assignedWorkerId) {
        const worker = workers.find((w) => w.id === task.assignedWorkerId);
        if (worker) workerMap.set(worker.id, worker);
      }
    });
    return Array.from(workerMap.values());
  }, [workers, planning]);

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

  return (
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
            Aujourd'hui
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
                const isToday =
                  day.toDateString() === new Date().toDateString();
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
                <tr key={worker.id} className="border-b border-zinc-200 dark:border-zinc-700">
                  <td className="sticky left-0 z-10 border-r border-zinc-200 bg-white px-4 py-3 text-sm font-medium text-zinc-900 dark:border-zinc-700 dark:bg-zinc-900 dark:text-white">
                    <div>{worker.name}</div>
                    <div className="text-xs text-zinc-500 dark:text-zinc-400">
                      {worker.role || 'Rôle non défini'}
                    </div>
                  </td>
                  {weekDays.map((day) => {
                    const dayKey = day.toISOString().split('T')[0];
                    const dayTasks = tasksByDayAndWorker[dayKey]?.[worker.id] || [];
                    const totalHours = dayTasks.reduce((sum, task) => sum + task.hours, 0);
                    const isToday =
                      day.toDateString() === new Date().toDateString();

                    return (
                      <td
                        key={day.toISOString()}
                        className={`min-w-[120px] border-r border-zinc-200 bg-white p-2 align-top last:border-r-0 dark:border-zinc-700 dark:bg-zinc-900 ${
                          isToday
                            ? 'bg-emerald-50/50 dark:bg-emerald-900/10'
                            : ''
                        }`}
                      >
                        {dayTasks.length > 0 ? (
                          <div className="space-y-1">
                            {dayTasks.map((task) => (
                              <div
                                key={task.taskId}
                                className={`rounded-lg border px-2 py-1.5 text-xs ${
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
                            ))}
                            {totalHours > 0 && (
                              <div className="mt-1 text-[10px] font-semibold text-zinc-500 dark:text-zinc-400">
                                Total: {totalHours}h
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
                  Aucun employé assigné. Ajoutez des membres à l'équipe pour voir le planning.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

