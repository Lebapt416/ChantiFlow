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
      className="border border-rule-soft px-4 py-2 text-sm font-medium text-ink-2 transition hover:text-ink hover:border-rule disabled:cursor-not-allowed disabled:opacity-70 dark:border-rule dark:text-ink-2"
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
      }
    }
  }, [state?.success, router, onSuccess]);

  return (
    <form
      id="create-site-form"
      action={formAction}
      className="space-y-4 rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink"
    >
      {/* Nom du chantier */}
      <div className="space-y-2">
        <label
          htmlFor="name"
          className="text-sm font-medium text-ink-2"
        >
          Nom du chantier
        </label>
        <input
          id="name"
          name="name"
          placeholder="Résidence Soleil"
          className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
          required
        />
      </div>

      {/* Deadline */}
      <div className="space-y-2">
        <label
          htmlFor="deadline"
          className="text-sm font-medium text-ink-2"
        >
          Deadline
        </label>
        <input
          id="deadline"
          name="deadline"
          type="date"
          className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
          required
        />
      </div>

      {/* Code postal du chantier */}
      <div className="space-y-2">
        <label
          htmlFor="postal_code"
          className="text-sm font-medium text-ink-2"
        >
          Code postal du chantier
        </label>
        <CityAutocomplete
          id="postal_code"
          name="postal_code"
          placeholder="Ex: 75001, 69001, 13001... ou tapez le nom de la ville"
          className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
        />
        <p className="text-xs text-ink-3">
          🌤️ Le code postal permet à l&apos;IA d&apos;optimiser le planning selon la météo locale (tous les codes postaux de France sont supportés)
        </p>
      </div>

      {/* Bouton de soumission */}
      <div className="flex items-center justify-between pt-2">
        <div className="text-xs text-ink-3">
          {state?.error && (
            <span className="text-danger">{state.error}</span>
          )}
          {state?.success && (
            <span className="text-orange dark:text-green">
              Chantier créé avec succès.
            </span>
          )}
        </div>
        <SubmitButton />
      </div>
    </form>
  );
}
