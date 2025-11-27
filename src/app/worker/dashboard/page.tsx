import Image from 'next/image';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { readWorkerSession } from '@/lib/worker-session';
import { WorkerNav } from '../components/worker-nav';

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

  const { data: assignedTasks } = await supabase
    .from('tasks')
    .select('id, title, status, planned_start, planned_end, created_at, site_id')
    .eq('assigned_worker_id', worker.id)
    .order('planned_start', { ascending: true });

  const siteIds = new Set<string>();
  if (worker.site_id) siteIds.add(worker.site_id);
  assignedTasks?.forEach((task) => {
    if (task.site_id) siteIds.add(task.site_id);
  });

  let sitesById = new Map<
    string,
    { id: string; name: string | null; deadline: string | null; postal_code: string | null; completed_at: string | null }
  >();
  if (siteIds.size > 0) {
    const { data: sitesData } = await supabase
      .from('sites')
      .select('id, name, deadline, postal_code, completed_at')
      .in('id', Array.from(siteIds));
    sitesById = new Map((sitesData ?? []).map((site) => [site.id, site]));
  }

  const siteSummaries = Array.from(siteIds).map((siteId) => {
    const site =
      sitesById.get(siteId) ?? { id: siteId, name: 'Chantier', deadline: null, postal_code: null, completed_at: null };
    const tasksForSite = assignedTasks?.filter((task) => task.site_id === siteId) ?? [];
    const nextTaskDate =
      tasksForSite
        .map((task) => task.planned_start || task.planned_end || task.created_at)
        .filter(Boolean)
        .sort()[0] ?? null;
    return {
      site,
      tasksCount: tasksForSite.length,
      completed: !!site.completed_at,
      nextTaskDate,
    };
  });

  const inProgressSites = siteSummaries.filter((summary) => !summary.completed);
  const completedSites = siteSummaries.filter((summary) => summary.completed);

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-50 via-white to-zinc-100 pb-32 dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950">
      <header className="border-b border-white/60 bg-white/80 px-4 py-6 backdrop-blur dark:border-zinc-900/60 dark:bg-zinc-900/80">
        <div className="mx-auto flex max-w-5xl flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-zinc-900 shadow-lg shadow-black/10 dark:bg-white">
              <Image src="/logo.svg" alt="ChantiFlow" width={32} height={32} priority className="h-8 w-8" />
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-zinc-500 dark:text-zinc-400">ChantiFlow</p>
              <h1 className="text-2xl font-semibold text-zinc-900 dark:text-white">Mes chantiers</h1>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Bonjour {worker.name || 'compagnon'} — voici tous les sites que vous avez scannés.
              </p>
            </div>
          </div>
          <Link
            href="/worker/scanner"
            className="inline-flex items-center gap-2 rounded-full bg-emerald-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-emerald-700"
          >
            Scanner un nouveau chantier
          </Link>
        </div>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-8">
        <section className="rounded-3xl border border-zinc-200 bg-white/90 p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900/90">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">En cours</p>
              <h2 className="text-2xl font-semibold text-zinc-900 dark:text-white">Chantiers actifs</h2>
            </div>
            <span className="rounded-full bg-zinc-100 px-4 py-1 text-xs font-semibold text-zinc-600 dark:bg-zinc-800 dark:text-zinc-200">
              {inProgressSites.length} actif(s)
            </span>
          </div>
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            {inProgressSites.length ? (
              inProgressSites.map(({ site, nextTaskDate, tasksCount }) => (
                <div
                  key={site.id}
                  className="flex flex-col gap-3 rounded-2xl border border-zinc-200 bg-zinc-50/80 p-4 dark:border-zinc-800 dark:bg-zinc-900/60"
                >
                  <div>
                    <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">{site.name || 'Chantier'}</h3>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      {tasksCount} tâche(s) assignée(s)
                    </p>
                  </div>
                  <p className="text-sm text-zinc-600 dark:text-zinc-300">
                    Prochaine mission : {nextTaskDate ? formatDate(nextTaskDate) : 'à définir'}
                  </p>
                </div>
              ))
            ) : (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Aucun chantier actif pour le moment. Scannez un QR code pour en ajouter un nouveau.
              </p>
            )}
          </div>
        </section>

        <section className="rounded-3xl border border-zinc-200 bg-white/90 p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900/90">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">Terminés</p>
              <h2 className="text-2xl font-semibold text-zinc-900 dark:text-white">Chantiers archivés</h2>
            </div>
            <span className="rounded-full bg-zinc-100 px-4 py-1 text-xs font-semibold text-zinc-600 dark:bg-zinc-800 dark:text-zinc-200">
              {completedSites.length} terminé(s)
            </span>
          </div>
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            {completedSites.length ? (
              completedSites.map(({ site }) => (
                <div
                  key={site.id}
                  className="rounded-2xl border border-zinc-200 bg-zinc-50/80 p-4 text-sm text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900/60 dark:text-zinc-300"
                >
                  <p className="text-base font-semibold text-zinc-900 dark:text-white">{site.name || 'Chantier'}</p>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    Terminé le {site.completed_at ? formatDate(site.completed_at) : 'date inconnue'}
                  </p>
                </div>
              ))
            ) : (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Aucun chantier clôturé pour l&apos;instant.
              </p>
            )}
          </div>
        </section>
      </main>

      <WorkerNav />
    </div>
  );

}

function formatDate(date?: string | null) {
  if (!date) return 'Non définie';
  try {
    return new Date(date).toLocaleDateString('fr-FR', {
      day: '2-digit',
      month: 'short',
    });
  } catch {
    return 'Non définie';
  }
}

