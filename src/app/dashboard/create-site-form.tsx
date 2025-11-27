"use client";

import { useActionState, useEffect } from "react";
import { useFormStatus } from "react-dom";
import { useRouter } from "next/navigation";
import { createSiteAction, type CreateSiteState } from './actions';
import { CityAutocomplete } from '@/components/city-autocomplete';

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

type Props = {
  onSuccess?: () => void;
};

export function CreateSiteForm({ onSuccess }: Props) {
  const router = useRouter();
  const [state, formAction] = useActionState(createSiteAction, initialState);

  useEffect(() => {
    if (state?.success) {
      const form = document.getElementById('create-site-form') as HTMLFormElement | null;
      form?.reset();
      if (onSuccess) {
        onSuccess();
      } else {
        router.push('/sites');
        router.refresh();
      }
    }
  }, [state?.success, router, onSuccess]);

  return (
    <form
      id="create-site-form"
      action={formAction}
      className="space-y-4 rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900"
    >
      {/* Nom du chantier */}
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

      {/* Deadline */}
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

      {/* Code postal du chantier */}
      <div className="space-y-2">
        <label
          htmlFor="postal_code"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Code postal du chantier
        </label>
        <CityAutocomplete
          id="postal_code"
          name="postal_code"
          placeholder="Ex: 75001, 69001, 13001... ou tapez le nom de la ville"
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
        />
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          🌤️ Le code postal permet à l&apos;IA d&apos;optimiser le planning selon la météo locale (tous les codes postaux de France sont supportés)
        </p>
      </div>

      {/* Bouton de soumission */}
      <div className="flex items-center justify-between pt-2">
        <div className="text-xs text-zinc-500 dark:text-zinc-400">
          {state?.error && (
            <span className="text-rose-600 dark:text-rose-400">{state.error}</span>
          )}
          {state?.success && (
            <span className="text-emerald-600 dark:text-emerald-400">
              Chantier créé avec succès.
            </span>
          )}
        </div>
        <SubmitButton />
      </div>
    </form>
  );
}

