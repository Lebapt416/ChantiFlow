'use client';

import { useActionState, useEffect } from 'react';
import { useFormStatus } from 'react-dom';
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

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
type Props = Record<string, never>;

export function AddWorkerForm({}: Props) {
  const [state, formAction] = useActionState(addWorkerAction, initialState);

  useEffect(() => {
    if (state?.success) {
      const form = document.getElementById('add-worker-form') as HTMLFormElement | null;
      form?.reset();
    }
  }, [state?.success]);

  return (
    <form
      id="add-worker-form"
      action={formAction}
      className="grid gap-4 rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 md:grid-cols-3 dark:border-zinc-800 dark:bg-zinc-900"
    >
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
          required
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
      <div className="md:col-span-3 flex items-end">
        <SubmitButton />
      </div>
      {state?.error ? (
        <p className="md:col-span-3 text-sm text-rose-400">{state.error}</p>
      ) : null}
      {state?.success ? (
        <p className="md:col-span-3 text-sm text-emerald-400">
          Membre ajouté à votre équipe. Vous pourrez l&apos;assigner à un chantier plus tard.
        </p>
      ) : null}
    </form>
  );
}

