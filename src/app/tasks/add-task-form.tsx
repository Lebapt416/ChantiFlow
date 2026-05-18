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
      className="rounded border border-rule-soft px-4 py-2 text-sm font-medium text-ink-2 transition hover:text-ink hover:border-rule disabled:cursor-not-allowed disabled:opacity-70 dark:border-rule dark:text-ink-2"
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
      <div className="rounded border border-dashed border-rule-soft p-6 text-center text-sm text-ink-3 dark:border-rule">
        Créez d&apos;abord un chantier pour ajouter des tâches.
      </div>
    );
  }

  return (
    <form
      id="add-task-form"
      action={formAction}
      className="grid gap-4 rounded border border-rule-soft bg-paper p-6 md:grid-cols-2 dark:border-rule dark:bg-ink"
    >
      <div className="space-y-2">
        <label
          htmlFor="siteId"
          className="text-sm font-medium text-ink-2"
        >
          Chantier
        </label>
        <select
          id="siteId"
          name="siteId"
          className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
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
          className="text-sm font-medium text-ink-2"
        >
          Titre
        </label>
        <input
          id="title"
          name="title"
          placeholder="Couler la dalle"
          className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
          required
        />
      </div>
      <div className="space-y-2">
        <label
          htmlFor="required_role"
          className="text-sm font-medium text-ink-2"
        >
          Rôle requis
        </label>
        <input
          id="required_role"
          name="required_role"
          placeholder="Maçon"
          className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
        />
      </div>
      <div className="space-y-2">
        <label
          htmlFor="duration_hours"
          className="text-sm font-medium text-ink-2"
        >
          Durée (heures)
        </label>
        <input
          id="duration_hours"
          name="duration_hours"
          type="number"
          min="1"
          className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
        />
      </div>
      <div className="flex items-end">
        <SubmitButton />
      </div>
      {state?.error ? (
        <p className="md:col-span-2 text-sm text-danger">{state.error}</p>
      ) : null}
      {state?.success ? (
        <p className="md:col-span-2 text-sm text-green">Tâche ajoutée.</p>
      ) : null}
    </form>
  );
}
