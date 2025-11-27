import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { readWorkerSession } from '@/lib/worker-session';
import { WorkerScanner } from './worker-scanner';
import { WorkerSiteShell } from '../components/worker-site-shell';

export const dynamic = 'force-dynamic';

export default async function WorkerDashboardPage() {
  const session = await readWorkerSession();
  if (!session?.workerId) {
    redirect('/worker/login');
  }

  const supabase = await createSupabaseServerClient();
  const { data: worker } = await supabase
    .from('workers')
    .select('id, name, site_id')
    .eq('id', session.workerId)
    .single();

  if (!worker) {
    redirect('/worker/login');
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-50 via-white to-zinc-100 dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950">
      <header className="border-b border-white/60 bg-white/70 px-4 py-4 backdrop-blur dark:border-zinc-900/60 dark:bg-zinc-900/70">
        <div className="mx-auto flex max-w-5xl flex-col gap-1">
          <p className="text-xs uppercase tracking-[0.4em] text-zinc-500 dark:text-zinc-400">ChantiFlow</p>
          <h1 className="text-2xl font-semibold text-zinc-900 dark:text-white">Espace Ouvrier</h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Bonjour {worker.name || 'compagnon'} — votre session reste ouverte même si vous fermez l&apos;application.
          </p>
        </div>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-8">
        <WorkerScanner />

        {worker.site_id ? (
          <section className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">Chantier actif</p>
                <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">Vos tâches en cours</h2>
              </div>
            </div>
            <WorkerSiteShell siteId={worker.site_id} />
          </section>
        ) : (
          <section className="rounded-3xl border border-dashed border-zinc-200 bg-white/80 p-6 text-center shadow-inner dark:border-zinc-800 dark:bg-zinc-950/40">
            <p className="text-lg font-semibold text-zinc-900 dark:text-white">Aucun chantier scanné</p>
            <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
              Utilisez le bouton ci-dessus pour scanner le QR code installé sur le chantier et accéder automatiquement à votre planning.
            </p>
          </section>
        )}
      </main>
    </div>
  );
}

