import Image from 'next/image';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { readWorkerSession } from '@/lib/worker-session';
import { WorkerNav } from '../components/worker-nav';
import { WorkerSiteShell } from '../components/worker-site-shell';

export const dynamic = 'force-dynamic';

export default async function WorkerReportsPage() {
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
    <div className="min-h-screen bg-gradient-to-b from-zinc-50 via-white to-zinc-100 pb-32 dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950">
      <header className="border-b border-white/80 bg-white/90 px-4 py-6 backdrop-blur dark:border-zinc-900/60 dark:bg-zinc-900/80">
        <div className="mx-auto flex max-w-5xl flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-zinc-900 shadow-lg shadow-black/10 dark:bg-white">
              <Image src="/logo.svg" alt="ChantiFlow" width={32} height={32} priority className="h-8 w-8" />
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-zinc-500 dark:text-zinc-400">ChantiFlow</p>
              <h1 className="text-2xl font-semibold text-zinc-900 dark:text-white">Rapports & remontées</h1>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Envoyez des photos, commentaires et incidents à votre chef de chantier.
              </p>
            </div>
          </div>
          <Link
            href="/worker/scanner"
            className="inline-flex items-center gap-2 rounded-full border border-zinc-300 px-4 py-2 text-xs font-semibold text-zinc-700 transition hover:border-zinc-400 hover:bg-white dark:border-zinc-700 dark:text-zinc-200 dark:hover:border-zinc-500"
          >
            Scanner un nouveau chantier
          </Link>
        </div>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-8">
        {worker.site_id ? (
          <WorkerSiteShell siteId={worker.site_id} defaultTab="reports" showNavigation={false} />
        ) : (
          <section className="rounded-3xl border border-dashed border-zinc-200 bg-white/80 p-6 text-center shadow-inner dark:border-zinc-800 dark:bg-zinc-950/40">
            <p className="text-lg font-semibold text-zinc-900 dark:text-white">Aucun chantier assigné</p>
            <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
              Rejoignez un chantier via QR code pour pouvoir envoyer des rapports et remontées.
            </p>
          </section>
        )}
      </main>

      <WorkerNav />
    </div>
  );
}

