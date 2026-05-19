'use client';

import { useActionState, useEffect } from 'react';
import { useFormStatus } from 'react-dom';
import { useRouter } from 'next/navigation';
import { createSiteAction, type CreateSiteState } from '@/app/dashboard/actions';
import { CityAutocomplete } from '@/components/city-autocomplete';

const initialState: CreateSiteState = {};

function SubmitButton() {
  const { pending } = useFormStatus();
  return (
    <button
      type="submit"
      disabled={pending}
      className="w-full flex items-center justify-between gap-3 px-6 py-4 bg-ink text-paper border border-ink hover:bg-orange hover:border-orange transition-colors duration-150 font-sans font-medium text-[15px] disabled:opacity-60 disabled:cursor-not-allowed"
    >
      <span>{pending ? 'Création en cours…' : 'Créer le chantier'}</span>
      <span className="font-mono text-[11px] opacity-55 tracking-widest">01 →</span>
    </button>
  );
}

type Props = {
  canCreate: boolean;
  limitReason?: string;
  currentCount: number;
  maxSites: number;
};

export function DashboardNewSiteForm({ canCreate, limitReason, currentCount, maxSites }: Props) {
  const router = useRouter();
  const [state, formAction] = useActionState(createSiteAction, initialState);

  useEffect(() => {
    if (state?.success) {
      router.refresh();
    }
  }, [state?.success, router]);

  if (!canCreate) {
    return (
      <div className="border border-warn bg-paper-2 p-6">
        <p className="font-mono text-[10px] uppercase tracking-widest text-warn mb-2">Limite de plan atteinte</p>
        <p className="text-[14px] text-ink-2 mb-4">
          {limitReason || `Vous avez atteint la limite de ${maxSites} chantier${maxSites > 1 ? 's' : ''}.`}
        </p>
        <a
          href="/account"
          className="inline-flex items-center gap-2 border border-ink px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-ink hover:bg-ink hover:text-paper transition-colors"
        >
          Passer au plan supérieur →
        </a>
      </div>
    );
  }

  return (
    <form id="dashboard-create-site-form" action={formAction}>
      {/* Nom */}
      <div className="mb-6">
        <label className="block font-mono text-[10px] uppercase tracking-widest text-ink-2 mb-2">
          Nom du chantier
        </label>
        <input
          name="name"
          type="text"
          placeholder="Résidence Soleil"
          required
          className="w-full px-4 py-3.5 bg-transparent border border-rule text-ink font-sans text-[15px] placeholder:text-ink-3 focus:outline-none focus:border-orange transition-colors"
        />
      </div>

      {/* Deadline */}
      <div className="mb-6">
        <label className="block font-mono text-[10px] uppercase tracking-widest text-ink-2 mb-2">
          Deadline
        </label>
        <input
          name="deadline"
          type="date"
          required
          className="w-full px-4 py-3.5 bg-transparent border border-rule text-orange font-mono text-[15px] tracking-widest focus:outline-none focus:border-orange transition-colors"
        />
      </div>

      {/* Code postal */}
      <div className="mb-8">
        <label className="block font-mono text-[10px] uppercase tracking-widest text-ink-2 mb-2">
          Code postal / Ville
        </label>
        <CityAutocomplete
          name="postal_code"
          id="postal_code"
          placeholder="Ex : 75001, 69001, 13001… ou nom de ville"
          className="w-full px-4 py-3.5 bg-transparent border border-rule text-ink font-sans text-[15px] placeholder:text-ink-3 focus:outline-none focus:border-orange transition-colors"
        />
        <p className="mt-2 font-mono text-[11px] text-ink-2 leading-relaxed tracking-[0.02em]">
          <span className="text-orange">↗</span> Le code postal permet à l&apos;IA d&apos;optimiser le planning selon la météo locale.
        </p>
      </div>

      {/* Error / success */}
      {state?.error && (
        <p className="mb-4 font-mono text-[11px] text-danger tracking-widest">{state.error}</p>
      )}
      {state?.success && (
        <p className="mb-4 font-mono text-[11px] text-orange tracking-widest">✓ Chantier créé avec succès.</p>
      )}

      {/* Séparateur + bouton */}
      <div className="pt-6 border-t border-rule-soft">
        <SubmitButton />
        <p className="mt-3 font-mono text-[10px] text-ink-3 tracking-widest">
          {currentCount}/{maxSites === Infinity ? '∞' : maxSites} chantier{maxSites > 1 ? 's' : ''} utilisé{maxSites > 1 ? 's' : ''}
        </p>
      </div>
    </form>
  );
}
