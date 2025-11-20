"use client";

import { useActionState, useEffect } from "react";
import { useFormStatus } from "react-dom";
import { createSiteAction, type CreateSiteState } from './actions';

const initialState: CreateSiteState = {};

function SubmitButton() {
  const { pending } = useFormStatus();

  return (
    <button
      type="submit"
      className="rounded-md bg-black px-4 py-2 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-70 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
      disabled={pending}
    >
      {pending ? 'Création...' : 'Créer'}
    </button>
  );
}

export function CreateSiteForm() {
  const [state, formAction] = useActionState(createSiteAction, initialState);

  useEffect(() => {
    if (state?.success) {
      const form = document.getElementById('create-site-form') as HTMLFormElement | null;
      form?.reset();
    }
  }, [state?.success]);

  return (
    <form
      id="create-site-form"
      action={formAction}
      className="grid gap-4 rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 md:grid-cols-[2fr_1fr_auto] dark:border-zinc-800 dark:bg-zinc-900"
    >
      <div className="space-y-2">
        <label
          htmlFor="name"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Nom du chantier
        </label>
        <input
          id="name"
          name="name"
          placeholder="Résidence Soleil"
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          required
        />
      </div>
      <div className="space-y-2">
        <label
          htmlFor="deadline"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Deadline
        </label>
        <input
          id="deadline"
          name="deadline"
          type="date"
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          required
        />
      </div>
      <div className="flex items-end">
        <SubmitButton />
      </div>
      {state?.error ? (
        <p className="md:col-span-3 text-sm text-rose-400">{state.error}</p>
      ) : null}
      {state?.success ? (
        <p className="md:col-span-3 text-sm text-emerald-400">
          Chantier créé avec succès.
        </p>
      ) : null}
    </form>
  );
}

