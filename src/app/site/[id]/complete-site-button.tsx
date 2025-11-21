'use client';

import { useFormStatus } from 'react-dom';
import { completeSiteAction } from './actions';
import { useState } from 'react';
import { useRouter } from 'next/navigation';

function CompleteSiteSubmit({ disabled }: { disabled?: boolean }) {
  const { pending } = useFormStatus();
  return (
    <button
      type="submit"
      className="rounded-full border border-rose-600 bg-rose-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-700 disabled:cursor-not-allowed disabled:opacity-60 dark:border-rose-500 dark:bg-rose-500 dark:hover:bg-rose-600"
      disabled={pending || disabled}
    >
      {pending ? 'Finalisation...' : 'Terminer le chantier'}
    </button>
  );
}

type Props = {
  siteId: string;
  siteName: string;
};

export function CompleteSiteButton({ siteId, siteName }: Props) {
  const [showConfirm, setShowConfirm] = useState(false);
  const router = useRouter();

  async function handleSubmit(formData: FormData) {
    const result = await completeSiteAction({}, formData);
    if (result.success) {
      router.push('/dashboard');
      router.refresh();
    }
  }

  if (showConfirm) {
    return (
      <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4 dark:border-rose-900/60 dark:bg-rose-900/20">
        <p className="text-sm font-semibold text-rose-900 dark:text-rose-200">
          ⚠️ Confirmer la finalisation
        </p>
        <p className="mt-2 text-xs text-rose-700 dark:text-rose-300">
          Cette action va terminer le chantier <strong>{siteName}</strong>, retirer tous les employés et leur envoyer un email de notification. Cette action est irréversible.
        </p>
        <div className="mt-4 flex gap-2">
          <form action={handleSubmit}>
            <input type="hidden" name="siteId" value={siteId} />
            <CompleteSiteSubmit />
          </form>
          <button
            type="button"
            onClick={() => setShowConfirm(false)}
            className="rounded-full border border-zinc-300 px-4 py-2 text-sm font-semibold text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-600 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            Annuler
          </button>
        </div>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={() => setShowConfirm(true)}
      className="rounded-full border border-rose-600 bg-rose-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-700 dark:border-rose-500 dark:bg-rose-500 dark:hover:bg-rose-600"
    >
      Terminer le chantier
    </button>
  );
}

