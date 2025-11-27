'use client';

import Link from 'next/link';
import { useMemo, useState } from 'react';
import { X, PlayCircle, PauseCircle, Calendar } from 'lucide-react';

type TaskMeta = {
  id: string;
  title: string | null;
  status: string | null;
  required_role: string | null;
  date: string | null;
  siteId: string | null;
  siteName: string | null;
  group: WorkerTaskGroup['key'];
};

export type WorkerTaskGroup = {
  key: 'todo' | 'progress' | 'done';
  label: string;
  description: string;
  accent: string;
  badge: string;
};

type Props = {
  tasks: TaskMeta[];
  groups: WorkerTaskGroup[];
};

export function WorkerTasksList({ tasks, groups }: Props) {
  const tasksByGroup = useMemo(() => {
    const map: Record<WorkerTaskGroup['key'], TaskMeta[]> = {
      todo: [],
      progress: [],
      done: [],
    };
    tasks.forEach((task) => {
      map[task.group].push(task);
    });
    return map;
  }, [tasks]);

  const [selectedTask, setSelectedTask] = useState<TaskMeta | null>(null);

  return (
    <>
      <section className="grid gap-4 md:grid-cols-3">
        {groups.map((group) => (
          <div
            key={group.key}
            className="rounded-3xl border border-zinc-200 bg-white/90 p-4 shadow-sm transition hover:shadow-lg dark:border-zinc-800 dark:bg-zinc-900/90"
          >
            <p className={`text-xs uppercase tracking-[0.3em] ${group.accent}`}>{group.label}</p>
            <p className="mt-2 text-3xl font-semibold text-zinc-900 dark:text-white">{tasksByGroup[group.key].length}</p>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">{group.description}</p>
          </div>
        ))}
      </section>

      {groups.map((group) => (
        <section
          key={group.key}
          className="rounded-3xl border border-zinc-200 bg-white/90 p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900/90"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className={`text-xs uppercase tracking-[0.3em] ${group.accent}`}>{group.label}</p>
              <h2 className="text-2xl font-semibold text-zinc-900 dark:text-white">
                {tasksByGroup[group.key].length} mission(s)
              </h2>
            </div>
            <span className={`rounded-full px-3 py-1 text-xs font-semibold ${group.badge}`}>{group.description}</span>
          </div>
          <div className="mt-6 space-y-4">
            {tasksByGroup[group.key].length ? (
              tasksByGroup[group.key].map((task) => (
                <button
                  key={task.id}
                  type="button"
                  onClick={() => setSelectedTask(task)}
                  className="w-full rounded-2xl border border-zinc-200 bg-zinc-50/70 px-4 py-3 text-left transition hover:border-emerald-300 dark:border-zinc-700 dark:bg-zinc-900/60"
                >
                  <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
                    <span>{task.siteName}</span>
                    <span>{task.date ? formatDate(task.date) : 'À planifier'}</span>
                  </div>
                  <p className="mt-1 text-sm font-semibold text-zinc-900 dark:text-white">{task.title || 'Tâche'}</p>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    {task.required_role ? `Rôle : ${task.required_role}` : 'Rôle non précisé'}
                  </p>
                </button>
              ))
            ) : (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">Aucune tâche dans cette catégorie.</p>
            )}
          </div>
        </section>
      ))}

      {selectedTask ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4 py-10">
          <div className="w-full max-w-lg rounded-3xl border border-zinc-700 bg-zinc-900/95 p-6 text-white shadow-2xl">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-zinc-400">{selectedTask.siteName}</p>
                <h3 className="text-2xl font-semibold">{selectedTask.title || 'Tâche'}</h3>
              </div>
              <button
                type="button"
                onClick={() => setSelectedTask(null)}
                className="rounded-full border border-white/20 p-2 text-zinc-300 transition hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="space-y-3 text-sm text-zinc-300">
              <p className="flex items-center gap-2">
                <Calendar className="h-4 w-4" />
                {selectedTask.date ? formatDate(selectedTask.date) : 'Date à confirmer'}
              </p>
              <p className="flex items-center gap-2">
                <PlayCircle className="h-4 w-4" />
                Statut : {selectedTask.status || 'Non défini'}
              </p>
              <p className="flex items-center gap-2">
                <PauseCircle className="h-4 w-4" />
                Rôle requis : {selectedTask.required_role || 'Non précisé'}
              </p>
            </div>

            <div className="mt-6 flex flex-col gap-3">
              {selectedTask.siteId ? (
                <Link
                  href={`/worker/${selectedTask.siteId}`}
                  className="inline-flex w-full items-center justify-center rounded-2xl bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-500"
                >
                  Ouvrir ce chantier
                </Link>
              ) : null}
              <button
                type="button"
                className="inline-flex w-full items-center justify-center rounded-2xl border border-white/20 px-4 py-2 text-sm font-semibold text-white transition hover:border-white/40"
                onClick={() => setSelectedTask(null)}
              >
                Fermer
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}

function formatDate(date?: string | null) {
  if (!date) return 'Non définie';
  try {
    return new Date(date).toLocaleDateString('fr-FR', { day: '2-digit', month: 'short' });
  } catch {
    return 'Non définie';
  }
}

