'use client';

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { UserCheck } from 'lucide-react';

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
        const supabase = createSupabaseBrowserClient();

        // Chercher un worker de test ou créer un worker de test pour ce chantier
        // On cherche d'abord s'il existe déjà un worker de test
        const { data: existingTestWorker, error: fetchError } = await supabase
          .from('workers')
          .select('id, name, email, access_code')
          .eq('site_id', siteId)
          .eq('email', 'test@chantiflow.com')
          .maybeSingle();

        if (fetchError && fetchError.code !== 'PGRST116') {
          console.error('Erreur recherche worker test:', fetchError);
        }

        let workerId: string;
        let accessCode: string;

        if (existingTestWorker) {
          // Utiliser le worker existant
          workerId = existingTestWorker.id;
          accessCode = existingTestWorker.access_code || 'TEST1234';
        } else {
          // Créer un worker de test
          const { generateAccessCode } = await import('@/lib/access-code');
          accessCode = generateAccessCode();

          // Essayer d'insérer avec access_code, sinon sans
          let insertData: any = {
            site_id: siteId,
            name: 'Test Employé',
            email: 'test@chantiflow.com',
            role: 'Test',
          };

          // Essayer d'ajouter access_code si la colonne existe
          try {
            insertData.access_code = accessCode;
          } catch (e) {
            // Ignorer si la colonne n'existe pas
          }

          const { data: newWorker, error: insertError } = await supabase
            .from('workers')
            .insert(insertData)
            .select('id')
            .single();

          if (insertError || !newWorker) {
            setError('Impossible de créer le worker de test. ' + (insertError?.message || ''));
            return;
          }

          workerId = newWorker.id;
        }

        // Stocker dans le localStorage pour simuler la connexion
        localStorage.setItem(`worker_${siteId}`, workerId);
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

