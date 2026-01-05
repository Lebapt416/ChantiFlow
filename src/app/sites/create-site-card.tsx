'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Plus, X } from 'lucide-react';
import { CreateSiteForm } from '../dashboard/create-site-form';
import Link from 'next/link';

type Props = {
  canCreate: boolean;
  limitReason?: string;
  currentCount: number;
  maxSites: number;
};

export function CreateSiteCard({ canCreate, limitReason, currentCount, maxSites }: Props) {
  const [isFormOpen, setIsFormOpen] = useState(false);
  const router = useRouter();

  function handleSuccess() {
    setIsFormOpen(false);
    // Recharger la page pour afficher le nouveau chantier
    router.push('/sites');
  }

  if (!canCreate) {
    return (
      <div className="flex-shrink-0 w-80 rounded-2xl border-2 border-dashed border-amber-300 bg-amber-50 p-6 shadow-lg shadow-black/5 dark:border-amber-700 dark:bg-amber-900/20">
        <div className="flex h-full flex-col items-center justify-center text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full border-2 border-amber-300 bg-white dark:border-amber-700 dark:bg-zinc-800">
            <Plus className="h-8 w-8 text-amber-500 dark:text-amber-400" />
          </div>
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
            Limite atteinte
          </h3>
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            {limitReason || `Vous avez atteint la limite de ${maxSites} chantier${maxSites > 1 ? 's' : ''}.`}
          </p>
          <Link
            href="/account"
            className="mt-4 rounded-lg border border-amber-300 bg-amber-100 px-4 py-2 text-xs font-semibold text-amber-900 transition hover:bg-amber-200 dark:border-amber-700 dark:bg-amber-900/40 dark:text-amber-200 dark:hover:bg-amber-900/60"
          >
            Passer au plan supérieur →
          </Link>
        </div>
      </div>
    );
  }

  if (isFormOpen) {
    return (
      <div className="flex-shrink-0 w-80 rounded-2xl border-2 border-zinc-300 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-700 dark:bg-zinc-900">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
            Nouveau chantier
          </h3>
          <button
            type="button"
            onClick={() => setIsFormOpen(false)}
            className="rounded-lg p-1 transition hover:bg-zinc-100 dark:hover:bg-zinc-800"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <CreateSiteForm onSuccess={handleSuccess} />
        <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
          {currentCount}/{maxSites === Infinity ? '∞' : maxSites} chantier{maxSites > 1 ? 's' : ''} utilisé{maxSites > 1 ? 's' : ''}
        </p>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={() => setIsFormOpen(true)}
      className="flex-shrink-0 w-80 rounded-2xl border-2 border-dashed border-zinc-300 bg-zinc-50 p-6 shadow-lg shadow-black/5 transition hover:border-zinc-400 hover:bg-zinc-100 hover:shadow-xl hover:scale-105 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:border-zinc-600"
    >
      <div className="flex h-full flex-col items-center justify-center text-center">
        <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full border-2 border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-800">
          <Plus className="h-8 w-8 text-zinc-500 dark:text-zinc-400" />
        </div>
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
          Nouveau chantier
        </h3>
        <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
          Créez un nouveau projet de construction
        </p>
        <p className="mt-1 text-xs text-zinc-400 dark:text-zinc-500">
          {currentCount}/{maxSites === Infinity ? '∞' : maxSites} utilisé{maxSites > 1 ? 's' : ''}
        </p>
      </div>
    </button>
  );
}

