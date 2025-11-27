'use client';

import { useState, useTransition } from 'react';
import { useFormStatus } from 'react-dom';
import Link from 'next/link';
import { UserPlus, X, Loader2, User } from 'lucide-react';
import { assignTaskAction, type AssignTaskState } from './actions';

type Worker = {
  id: string;
  name: string;
  email: string | null;
  role: string | null;
};

type Props = {
  taskId: string;
  siteId: string;
  currentWorkerId: string | null;
  availableWorkers: Worker[];
};

function SubmitButton({ isPending }: { isPending: boolean }) {
  return (
    <button
      type="submit"
      disabled={isPending}
      className="rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {isPending ? (
        <Loader2 className="h-3 w-3 animate-spin" />
      ) : (
        'Assigner'
      )}
    </button>
  );
}

export function AssignTaskButton({ taskId, siteId, currentWorkerId, availableWorkers }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const [isPending, startTransition] = useTransition();
  const [state, setState] = useState<AssignTaskState | null>(null);

  const currentWorker = currentWorkerId
    ? availableWorkers.find((w) => w.id === currentWorkerId)
    : null;

  async function handleSubmit(formData: FormData) {
    startTransition(async () => {
      const result = await assignTaskAction({}, formData);
      setState(result);
      if (result.success) {
        setIsOpen(false);
        // Réinitialiser l'état après un court délai
        setTimeout(() => setState(null), 2000);
      }
    });
  }

  // Toujours afficher le bouton, même s'il n'y a pas de workers
  // Debug: afficher les infos en console
  if (typeof window !== 'undefined') {
    console.log('[AssignTaskButton]', {
      taskId,
      currentWorkerId,
      availableWorkersCount: availableWorkers.length,
      hasCurrentWorker: !!currentWorker,
    });
  }

  return (
    <div className="relative inline-block z-10">
      {currentWorker ? (
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 rounded-md bg-emerald-100 px-2 py-1 text-xs text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
            <User className="h-3 w-3" />
            <span className="font-medium">{currentWorker.name}</span>
            {currentWorker.role && (
              <span className="text-emerald-600 dark:text-emerald-500">
                ({currentWorker.role})
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={() => setIsOpen(!isOpen)}
            className="rounded-md p-1.5 text-xs text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
            title="Modifier l'assignation"
          >
            <UserPlus className="h-4 w-4" />
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-1.5 rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition hover:bg-zinc-50 hover:border-zinc-400 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700 dark:hover:border-zinc-500"
          title="Assigner cette tâche à un membre de l'équipe"
        >
          <UserPlus className="h-4 w-4" />
          <span>Assigner</span>
        </button>
      )}

      {isOpen && (
        <div className="absolute right-0 top-full z-10 mt-2 w-64 rounded-lg border border-zinc-200 bg-white p-3 shadow-lg dark:border-zinc-700 dark:bg-zinc-800">
          <div className="mb-2 flex items-center justify-between">
            <h4 className="text-xs font-semibold text-zinc-900 dark:text-white">
              Assigner à
            </h4>
            <button
              type="button"
              onClick={() => {
                setIsOpen(false);
                setState(null);
              }}
              className="rounded p-0.5 text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-700 dark:hover:text-zinc-300"
            >
              <X className="h-3 w-3" />
            </button>
          </div>

          {availableWorkers.length === 0 ? (
            <div className="space-y-2">
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Aucun membre d&apos;équipe disponible. Ajoutez des membres dans la page &quot;Équipe&quot;.
              </p>
              <Link
                href="/team"
                className="block rounded-md bg-emerald-600 px-3 py-1.5 text-center text-xs font-medium text-white transition hover:bg-emerald-700"
              >
                Aller à l&apos;équipe →
              </Link>
            </div>
          ) : (
            <form action={handleSubmit}>
              <input type="hidden" name="taskId" value={taskId} />
              
              <div className="space-y-2">
                <select
                  name="workerId"
                  className="w-full rounded-md border border-zinc-200 bg-white px-2 py-1.5 text-xs shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-900 dark:text-white"
                  defaultValue={currentWorkerId || ''}
                >
                  <option value="">Aucun (désassigner)</option>
                  {availableWorkers.map((worker) => (
                    <option key={worker.id} value={worker.id}>
                      {worker.name} {worker.role ? `(${worker.role})` : ''}
                    </option>
                  ))}
                </select>

                <div className="flex flex-col gap-2">
                  {state?.error && (
                    <div className="rounded-md bg-rose-50 p-2 text-xs text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">
                      {state.error}
                      {state.error.includes('assigned_worker_id') && (
                        <div className="mt-1">
                          <Link
                            href="/team"
                            className="underline"
                          >
                            Exécutez la migration SQL: migration-task-assigned-worker.sql
                          </Link>
                        </div>
                      )}
                    </div>
                  )}
                  {state?.success && (
                    <div className="rounded-md bg-emerald-50 p-2 text-xs text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300">
                      ✅ Assignation mise à jour avec succès
                    </div>
                  )}
                  <div className="flex items-center justify-end">
                    <SubmitButton isPending={isPending} />
                  </div>
                </div>
              </div>
            </form>
          )}
        </div>
      )}
    </div>
  );
}

