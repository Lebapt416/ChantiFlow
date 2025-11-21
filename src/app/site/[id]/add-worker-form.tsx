"use client";

import { useState, useActionState, useEffect } from "react";
import { useFormStatus } from "react-dom";
import { addWorkerAction, type ActionState } from './actions';

const initialState: ActionState = {};

function SubmitButton() {
  const { pending } = useFormStatus();
  return (
    <button
      type="submit"
      className="rounded-md bg-black px-4 py-2 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-70 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
      disabled={pending}
    >
      {pending ? 'Ajout...' : 'Ajouter'}
    </button>
  );
}

type Worker = {
  id: string;
  name: string;
  email: string | null;
  role: string | null;
};

type Props = {
  siteId: string;
  availableWorkers: Worker[];
};

export function AddWorkerForm({ siteId, availableWorkers }: Props) {
  const [useExisting, setUseExisting] = useState(false);
  const [state, formAction] = useActionState(addWorkerAction, initialState);

  useEffect(() => {
    if (state?.success) {
      const form = document.getElementById('add-worker-form') as HTMLFormElement | null;
      form?.reset();
      setUseExisting(false);
    }
  }, [state?.success]);

  return (
    <form
      id="add-worker-form"
      action={formAction}
      className="space-y-4 rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900"
    >
      <input type="hidden" name="siteId" value={siteId} />
      
      {availableWorkers.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="useExisting"
              checked={useExisting}
              onChange={(e) => setUseExisting(e.target.checked)}
              className="h-4 w-4 rounded border-zinc-300 text-black focus:ring-2 focus:ring-black/60 dark:border-zinc-600"
            />
            <label
              htmlFor="useExisting"
              className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
            >
              Sélectionner un membre existant de mon équipe
            </label>
          </div>
        </div>
      )}

      {useExisting && availableWorkers.length > 0 ? (
        <div className="space-y-2">
          <label
            htmlFor="existingWorkerId"
            className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
          >
            Membre de l'équipe
          </label>
          <select
            id="existingWorkerId"
            name="existingWorkerId"
            className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
            required
          >
            <option value="">-- Sélectionner un membre --</option>
            {availableWorkers.map((worker) => (
              <option key={worker.id} value={worker.id}>
                {worker.name} {worker.role ? `(${worker.role})` : ''} {worker.email ? `- ${worker.email}` : ''}
              </option>
            ))}
          </select>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-3">
          <div className="space-y-2">
            <label
              htmlFor="name"
              className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
            >
              Nom
            </label>
            <input
              id="name"
              name="name"
              placeholder="Camille Dupont"
              className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
              required={!useExisting}
            />
          </div>
          <div className="space-y-2">
            <label
              htmlFor="email"
              className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
            >
              Email
            </label>
            <input
              id="email"
              name="email"
              type="email"
              placeholder="camille@chantier.com"
              className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
            />
          </div>
          <div className="space-y-2">
            <label
              htmlFor="role"
              className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
            >
              Métier
            </label>
            <input
              id="role"
              name="role"
              placeholder="Chef d'équipe"
              className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
            />
          </div>
        </div>
      )}

      <div className="flex items-end">
        <SubmitButton />
      </div>
      {state?.error ? (
        <p className="text-sm text-rose-400">{state.error}</p>
      ) : null}
      {state?.success ? (
        <p className="text-sm text-emerald-400">
          Employé ajouté au chantier.
        </p>
      ) : null}
    </form>
  );
}

