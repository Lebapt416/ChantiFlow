'use client';

import { useState, useEffect } from 'react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { Clock, CheckCircle2 } from 'lucide-react';
import { TaskDetailModal } from './task-detail-modal';

type Task = {
  id: string;
  title: string;
  status: string;
  required_role: string | null;
};

type Props = {
  siteId: string;
  workerId: string;
};

export function TasksTab({ siteId, workerId }: Props) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [siteName, setSiteName] = useState('');
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      const supabase = createSupabaseBrowserClient();

      // Charger le nom du chantier
      const { data: site } = await supabase
        .from('sites')
        .select('name')
        .eq('id', siteId)
        .single();

      if (site) {
        setSiteName(site.name);
      }

      // Charger les tâches
      const { data: tasksData } = await supabase
        .from('tasks')
        .select('id, title, status, required_role')
        .eq('site_id', siteId)
        .order('created_at', { ascending: true });

      if (tasksData) {
        setTasks(tasksData);
      }

      setIsLoading(false);
    }

    loadData();
  }, [siteId]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-zinc-300 border-t-emerald-600"></div>
      </div>
    );
  }

  const pendingTasks = tasks.filter((t) => t.status !== 'done');
  const completedTasks = tasks.filter((t) => t.status === 'done');

  return (
    <div className="p-3 sm:p-4 md:p-6 space-y-4 sm:space-y-6">
      <div>
        <h2 className="text-base sm:text-lg font-semibold text-zinc-900 dark:text-white mb-1">
          {siteName || 'Chantier'}
        </h2>
        <p className="text-xs sm:text-sm text-zinc-600 dark:text-zinc-400">
          {pendingTasks.length} tâche{pendingTasks.length > 1 ? 's' : ''} en attente
        </p>
      </div>

      {pendingTasks.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-3">
            À faire
          </h3>
          <div className="space-y-2 sm:space-y-3">
            {pendingTasks.map((task) => (
              <button
                key={task.id}
                onClick={() => setSelectedTaskId(task.id)}
                className="w-full text-left rounded-lg border border-zinc-200 bg-white p-3 sm:p-4 transition hover:bg-zinc-50 hover:border-zinc-300 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:bg-zinc-800 dark:hover:border-zinc-700"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm sm:text-base text-zinc-900 dark:text-white break-words">
                      {task.title}
                    </p>
                    {task.required_role && (
                      <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                        {task.required_role}
                      </p>
                    )}
                  </div>
                  <Clock className="h-4 w-4 sm:h-5 sm:w-5 text-amber-500 flex-shrink-0" />
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {completedTasks.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-3">
            Terminées
          </h3>
          <div className="space-y-2">
            {completedTasks.map((task) => (
              <button
                key={task.id}
                onClick={() => setSelectedTaskId(task.id)}
                className="w-full text-left rounded-lg border border-zinc-200 bg-white p-4 opacity-75 transition hover:opacity-100 hover:bg-zinc-50 hover:border-zinc-300 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:bg-zinc-800 dark:hover:border-zinc-700"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="font-medium text-zinc-900 dark:text-white line-through">
                      {task.title}
                    </p>
                    {task.required_role && (
                      <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                        {task.required_role}
                      </p>
                    )}
                  </div>
                  <CheckCircle2 className="h-5 w-5 text-emerald-500 flex-shrink-0 ml-2" />
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {tasks.length === 0 && (
        <div className="text-center py-12">
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Aucune tâche disponible pour le moment.
          </p>
        </div>
      )}

      {/* Modal de détails */}
      {selectedTaskId && (
        <TaskDetailModal
          taskId={selectedTaskId}
          siteId={siteId}
          isOpen={!!selectedTaskId}
          onClose={() => setSelectedTaskId(null)}
        />
      )}
    </div>
  );
}

