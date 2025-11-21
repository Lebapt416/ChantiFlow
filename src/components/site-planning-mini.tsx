'use client';

import Link from 'next/link';
import { Calendar, Clock } from 'lucide-react';

type Site = {
  id: string;
  name: string;
  deadline: string | null;
};

type PlanningTask = {
  taskId: string;
  taskTitle: string;
  startDate: string;
  endDate: string;
  assignedWorkerId: string | null;
};

type Props = {
  site: Site;
  planning: PlanningTask[];
  workerCount: number;
  taskCount: number;
};

export function SitePlanningMini({ site, planning, workerCount, taskCount }: Props) {
  // Calculer l'occupation (nombre de jours avec des tâches / nombre total de jours)
  const calculateOccupation = () => {
    if (planning.length === 0) return 0;
    
    const uniqueDays = new Set<string>();
    planning.forEach((task) => {
      const start = new Date(task.startDate);
      const end = new Date(task.endDate);
      const current = new Date(start);
      
      while (current <= end) {
        uniqueDays.add(current.toISOString().split('T')[0]);
        current.setDate(current.getDate() + 1);
      }
    });
    
    // Calculer sur une période de 30 jours
    const totalDays = 30;
    return Math.round((uniqueDays.size / totalDays) * 100);
  };

  const occupation = calculateOccupation();

  return (
    <Link
      href={`/planning?site=${site.id}`}
      className="block rounded-2xl border border-zinc-200 bg-white p-4 transition hover:border-zinc-900 hover:shadow-lg dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-white"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
            {site.name}
          </h3>
          {site.deadline && (
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
              Deadline: {new Date(site.deadline).toLocaleDateString('fr-FR')}
            </p>
          )}
        </div>
        <Calendar className="h-5 w-5 text-zinc-400 flex-shrink-0" />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-zinc-500 dark:text-zinc-400">Occupation</span>
          <span className="font-semibold text-zinc-900 dark:text-white">{occupation}%</span>
        </div>
        <div className="h-2 rounded-full bg-zinc-200 dark:bg-zinc-800 overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all"
            style={{ width: `${Math.min(occupation, 100)}%` }}
          />
        </div>
        <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400 pt-1">
          <span className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {taskCount} tâche{taskCount > 1 ? 's' : ''}
          </span>
          <span>{workerCount} employé{workerCount > 1 ? 's' : ''}</span>
        </div>
      </div>
    </Link>
  );
}

