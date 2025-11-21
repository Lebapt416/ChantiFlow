'use client';

import { useState, useEffect, useTransition } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { isValidAccessCode } from '@/lib/access-code';

export default function VerifyCodePage() {
  const params = useParams();
  const siteId = params?.siteId as string;
  const [code, setCode] = useState('');
  const [error, setError] = useState('');
  const [isPending, startTransition] = useTransition();
  const router = useRouter();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError('');

    if (!code || !isValidAccessCode(code)) {
      setError('Code invalide. Format attendu : 4 lettres + 4 chiffres (ex: ABCD1234)');
      return;
    }

    if (!siteId || typeof siteId !== 'string') {
      setError('Site non trouvé');
      return;
    }

    startTransition(async () => {
      const supabase = createSupabaseBrowserClient();
      
      // Vérifier le code d'accès
      const { data: worker, error: workerError } = await supabase
        .from('workers')
        .select('id, name, email, role, site_id')
        .eq('access_code', code.toUpperCase())
        .eq('site_id', siteId)
        .maybeSingle();

      if (workerError || !worker) {
        setError('Code d\'accès invalide ou worker non trouvé pour ce chantier.');
        return;
      }

      // Stocker l'ID du worker dans le localStorage pour la session
      localStorage.setItem(`worker_${siteId}`, worker.id);
      localStorage.setItem(`worker_name_${siteId}`, worker.name || '');

      // Rediriger vers l'interface mobile
      router.push(`/worker/${siteId}`);
    });
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-100 via-white to-zinc-50 p-6 text-zinc-900 dark:from-zinc-900 dark:via-zinc-800 dark:to-black dark:text-white flex items-center justify-center">
      <div className="w-full max-w-md">
        <div className="rounded-3xl border border-zinc-200 bg-white/80 p-8 backdrop-blur dark:border-white/10 dark:bg-white/5 shadow-xl">
          <h1 className="text-3xl font-bold text-center mb-2 dark:text-white">
            Accès chantier
          </h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-400 text-center mb-8">
            Entrez votre code d'accès unique
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="code" className="block text-sm font-semibold text-zinc-800 dark:text-zinc-100 mb-2">
                Code d'accès
              </label>
              <input
                type="text"
                id="code"
                value={code}
                onChange={(e) => {
                  const value = e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '');
                  if (value.length <= 8) {
                    setCode(value);
                  }
                }}
                placeholder="ABCD1234"
                maxLength={8}
                className="w-full rounded-lg border border-zinc-300 bg-white px-4 py-3 text-center text-2xl font-mono font-bold tracking-widest text-zinc-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-800 dark:text-white"
                required
                autoFocus
              />
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 text-center">
                Format : 4 lettres + 4 chiffres
              </p>
            </div>

            {error && (
              <div className="rounded-lg bg-rose-50 p-3 text-sm text-rose-800 dark:bg-rose-900/30 dark:text-rose-300">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={isPending || code.length !== 8}
              className="w-full rounded-lg bg-emerald-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {isPending ? 'Vérification...' : 'Accéder'}
            </button>
          </form>

          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-6 text-center">
            Vous avez reçu ce code par email lors de votre inscription au chantier.
          </p>
        </div>
      </div>
    </div>
  );
}

