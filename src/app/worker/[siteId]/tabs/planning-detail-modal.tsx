'use client';

import { X, Calendar, Clock, User, AlertCircle, CheckCircle2 } from 'lucide-react';

type PlanningTask = {
  taskId: string;
  taskTitle: string;
  order: number;
  startDate: string;
  endDate: string;
  assignedWorkerId: string | null;
  assignedWorkerIds?: string[];
  priority: 'high' | 'medium' | 'low';
  estimatedHours?: number;
  validated?: boolean;
};

type Props = {
  planning: PlanningTask[];
  siteName: string;
  isOpen: boolean;
  onClose: () => void;
};

export function PlanningDetailModal({ planning, siteName, isOpen, onClose }: Props) {
  if (!isOpen) return null;

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('fr-FR', {
      weekday: 'long',
      day: 'numeric',
      month: 'long',
      year: 'numeric',
    });
  };

  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleTimeString('fr-FR', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'border-rose-200 bg-rose-50 text-rose-900 dark:border-rose-900/60 dark:bg-rose-900/20 dark:text-rose-200';
      case 'medium':
        return 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900/60 dark:bg-amber-900/20 dark:text-amber-200';
      default:
        return 'border-zinc-200 bg-zinc-50 text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300';
    }
  };

  const getPriorityLabel = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'Priorité haute';
      case 'medium':
        return 'Priorité moyenne';
      default:
        return 'Priorité basse';
    }
  };

  // Calculer la durée totale du planning
  const sortedPlanning = [...planning].sort((a, b) => {
    const dateA = new Date(a.startDate).getTime();
    const dateB = new Date(b.startDate).getTime();
    return dateA - dateB;
  });

  const startDate = sortedPlanning[0]?.startDate;
  const endDate = sortedPlanning[sortedPlanning.length - 1]?.endDate;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-2xl border border-zinc-200 bg-white shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
          <div>
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Planning détaillé
            </h2>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              {siteName} • {planning.length} tâche{planning.length > 1 ? 's' : ''}
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 sm:p-6 space-y-4">
          {/* Vue d'ensemble */}
          {startDate && endDate && (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 dark:border-emerald-900/60 dark:bg-emerald-900/20">
              <div className="flex items-center gap-2 mb-2">
                <Calendar className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                <h3 className="text-sm font-semibold text-emerald-900 dark:text-emerald-200">
                  Période du planning
                </h3>
              </div>
              <p className="text-sm text-emerald-800 dark:text-emerald-300">
                Du {formatDate(startDate)} au {formatDate(endDate)}
              </p>
            </div>
          )}

          {/* Liste des tâches avec horaires */}
          <div className="space-y-3">
            {sortedPlanning.map((task, index) => (
              <div
                key={task.taskId}
                className={`rounded-lg border p-4 ${getPriorityColor(task.priority)}`}
              >
                <div className="flex items-start justify-between gap-4 mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="flex items-center justify-center h-6 w-6 rounded-full bg-white dark:bg-zinc-800 text-xs font-bold text-zinc-700 dark:text-zinc-300">
                        {index + 1}
                      </span>
                      <h3 className="font-semibold text-base text-zinc-900 dark:text-white">
                        {task.taskTitle}
                      </h3>
                      {task.validated && (
                        <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                      )}
                    </div>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="px-2 py-0.5 rounded-full bg-white dark:bg-zinc-800 text-xs font-medium">
                        {getPriorityLabel(task.priority)}
                      </span>
                      <span className="text-xs text-zinc-600 dark:text-zinc-400">
                        Ordre: {task.order}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Horaires détaillés */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3 pt-3 border-t border-zinc-200 dark:border-zinc-700">
                  <div className="flex items-start gap-2">
                    <Calendar className="h-4 w-4 text-zinc-500 dark:text-zinc-400 mt-0.5" />
                    <div>
                      <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-0.5">
                        Date de début
                      </p>
                      <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                        {formatDate(task.startDate)}
                      </p>
                      <p className="text-xs text-zinc-500 dark:text-zinc-400">
                        {formatTime(task.startDate)}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-2">
                    <Calendar className="h-4 w-4 text-zinc-500 dark:text-zinc-400 mt-0.5" />
                    <div>
                      <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-0.5">
                        Date de fin
                      </p>
                      <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                        {formatDate(task.endDate)}
                      </p>
                      <p className="text-xs text-zinc-500 dark:text-zinc-400">
                        {formatTime(task.endDate)}
                      </p>
                    </div>
                  </div>

                  {task.estimatedHours && (
                    <div className="flex items-start gap-2">
                      <Clock className="h-4 w-4 text-zinc-500 dark:text-zinc-400 mt-0.5" />
                      <div>
                        <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-0.5">
                          Durée estimée
                        </p>
                        <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                          {task.estimatedHours} heure{task.estimatedHours > 1 ? 's' : ''}
                        </p>
                      </div>
                    </div>
                  )}

                  {(task.assignedWorkerId || task.assignedWorkerIds) && (
                    <div className="flex items-start gap-2">
                      <User className="h-4 w-4 text-zinc-500 dark:text-zinc-400 mt-0.5" />
                      <div>
                        <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-0.5">
                          Assigné à
                        </p>
                        <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                          {task.assignedWorkerIds && task.assignedWorkerIds.length > 1
                            ? `${task.assignedWorkerIds.length} personnes`
                            : 'Vous'}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

