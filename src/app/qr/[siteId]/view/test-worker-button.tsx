'use client';

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { UserCheck } from 'lucide-react';
import { createTestWorkerAction } from './actions';

type Props = {
  siteId: string;
  siteName: string;
};

export function TestWorkerButton({ siteId, siteName }: Props) {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);

  const handleTest = async () => {
    setError(null);

    startTransition(async () => {
      try {
        // Créer ou récupérer le worker de test via l'action serveur
        const result = await createTestWorkerAction(siteId);

        if (!result.success || !result.workerId) {
          setError(result.error || 'Impossible de créer le worker de test.');
          return;
        }

        // Stocker dans le localStorage pour simuler la connexion
        localStorage.setItem(`worker_${siteId}`, result.workerId);
        localStorage.setItem(`worker_name_${siteId}`, 'Test Employé');
        localStorage.setItem(`test_mode_${siteId}`, 'true');

        // Rediriger vers l'interface employé
        router.push(`/worker/${siteId}`);
      } catch (err) {
        console.error('Erreur test interface:', err);
        setError('Erreur lors du test. Veuillez réessayer.');
      }
    });
  };

  return (
    <div>
      <button
        onClick={handleTest}
        disabled={isPending}
        className="w-full flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-70"
      >
        <UserCheck className="h-5 w-5" />
        {isPending ? 'Chargement...' : 'Test interface employé'}
      </button>
      {error && (
        <p className="mt-2 text-xs text-rose-600 dark:text-rose-400">{error}</p>
      )}
    </div>
  );
}

