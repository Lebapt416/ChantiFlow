'use client';

import { useActionState, useEffect } from 'react';
import { useFormStatus } from 'react-dom';
import { addTaskAction, type ActionState } from './actions';

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

type Site = {
  id: string;
  name: string;
};

type Props = {
  sites: Site[];
};

export function AddTaskForm({ sites }: Props) {
  const [state, formAction] = useActionState(addTaskAction, initialState);

  useEffect(() => {
    if (state?.success) {
      const form = document.getElementById('add-task-form') as HTMLFormElement | null;
      form?.reset();
    }
  }, [state?.success]);

  if (sites.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-zinc-200 p-6 text-center text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
        Créez d'abord un chantier pour ajouter des tâches.
      </div>
    );
  }

  return (
    <form
      id="add-task-form"
      action={formAction}
      className="grid gap-4 rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 md:grid-cols-2 dark:border-zinc-800 dark:bg-zinc-900"
    >
      <div className="space-y-2">
        <label
          htmlFor="siteId"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Chantier
        </label>
        <select
          id="siteId"
          name="siteId"
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          required
        >
          <option value="">-- Sélectionner un chantier --</option>
          {sites.map((site) => (
            <option key={site.id} value={site.id}>
              {site.name}
            </option>
          ))}
        </select>
      </div>
      <div className="space-y-2">
        <label
          htmlFor="title"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Titre
        </label>
        <input
          id="title"
          name="title"
          placeholder="Couler la dalle"
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          required
        />
      </div>
      <div className="space-y-2">
        <label
          htmlFor="required_role"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Rôle requis
        </label>
        <input
          id="required_role"
          name="required_role"
          placeholder="Maçon"
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
        />
      </div>
      <div className="space-y-2">
        <label
          htmlFor="duration_hours"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Durée (heures)
        </label>
        <input
          id="duration_hours"
          name="duration_hours"
          type="number"
          min="1"
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
        />
      </div>
      <div className="flex items-end">
        <SubmitButton />
      </div>
      {state?.error ? (
        <p className="md:col-span-2 text-sm text-rose-400">{state.error}</p>
      ) : null}
      {state?.success ? (
        <p className="md:col-span-2 text-sm text-emerald-400">Tâche ajoutée.</p>
      ) : null}
    </form>
  );
}

