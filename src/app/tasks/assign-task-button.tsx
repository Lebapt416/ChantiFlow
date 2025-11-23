'use client';

import { useState, useTransition } from 'react';
import { useFormStatus } from 'react-dom';
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

  return (
    <div className="relative">
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
            className="rounded-md p-1 text-xs text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
          >
            <UserPlus className="h-3 w-3" />
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-1.5 rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs font-medium text-zinc-700 transition hover:bg-zinc-50 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
        >
          <UserPlus className="h-3 w-3" />
          Assigner
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

              <div className="flex items-center justify-between gap-2">
                <div className="text-xs text-zinc-500 dark:text-zinc-400">
                  {state?.error && (
                    <span className="text-rose-600 dark:text-rose-400">{state.error}</span>
                  )}
                  {state?.success && (
                    <span className="text-emerald-600 dark:text-emerald-400">
                      Assignation mise à jour
                    </span>
                  )}
                </div>
                <SubmitButton isPending={isPending} />
              </div>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}

