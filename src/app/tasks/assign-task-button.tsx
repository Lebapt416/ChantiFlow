'use client';

import { useState, useTransition } from 'react';
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
      className="border border-orange px-3 py-1.5 font-mono text-[10px] uppercase tracking-widest text-orange transition hover:bg-paper-2 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {isPending ? (
        <Loader2 className="h-3 w-3 animate-spin" />
      ) : (
        'Assigner'
      )}
    </button>
  );
}

export function AssignTaskButton({ taskId, currentWorkerId, availableWorkers }: Props) {
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
        setTimeout(() => setState(null), 2000);
      }
    });
  }

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
          <div className="flex items-center gap-1.5 border border-rule-soft bg-paper-2 px-2 py-1 font-mono text-[10px] text-ink">
            <User className="h-3 w-3" />
            <span className="font-medium">{currentWorker.name}</span>
            {currentWorker.role && (
              <span className="text-orange">
                ({currentWorker.role})
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={() => setIsOpen(!isOpen)}
            className="p-1.5 text-ink-3 transition hover:text-ink"
            title="Modifier l'assignation"
          >
            <UserPlus className="h-4 w-4" />
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-1.5 border border-rule-soft px-3 py-1.5 font-mono text-[10px] uppercase tracking-widest text-ink-2 transition hover:text-ink hover:border-rule dark:border-rule dark:text-ink-2"
          title="Assigner cette tâche à un membre de l'équipe"
        >
          <UserPlus className="h-4 w-4" />
          <span>Assigner</span>
        </button>
      )}

      {isOpen && (
        <div className="absolute right-0 top-full z-10 mt-2 w-64 border border-rule-soft bg-paper p-3 dark:border-rule dark:bg-ink">
          <div className="mb-2 flex items-center justify-between">
            <h4 className="font-mono text-[10px] uppercase tracking-widest text-ink-3">
              Assigner à
            </h4>
            <button
              type="button"
              onClick={() => {
                setIsOpen(false);
                setState(null);
              }}
              className="p-0.5 text-ink-3 transition hover:text-ink"
            >
              <X className="h-3 w-3" />
            </button>
          </div>

          {availableWorkers.length === 0 ? (
            <div className="space-y-2">
              <p className="text-xs text-ink-3">
                Aucun membre d&apos;équipe disponible. Ajoutez des membres dans la page &quot;Équipe&quot;.
              </p>
              <Link
                href="/team"
                className="block border border-orange px-3 py-1.5 text-center font-mono text-[10px] uppercase tracking-widest text-orange transition hover:bg-paper-2"
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
                  className="w-full rounded border border-rule-soft bg-paper px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-orange text-ink dark:border-rule dark:bg-ink dark:text-paper"
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
                    <div className="border border-danger p-2 font-mono text-[10px] text-danger">
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
                    <div className="border border-rule-soft bg-paper-2 p-2 font-mono text-[10px] text-ink">
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
