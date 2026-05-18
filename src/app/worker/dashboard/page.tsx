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
    <div className="min-h-screen bg-paper pb-32 dark:bg-ink">
      <header className="border-b border-rule-soft bg-paper px-4 py-6 dark:border-rule dark:bg-ink">
        <div className="mx-auto flex max-w-5xl flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded bg-ink dark:bg-paper">
              <Image src="/logo.svg" alt="ChantiFlow" width={32} height={32} priority className="h-8 w-8" />
            </div>
            <div>
              <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3">ChantiFlow</p>
              <h1 className="font-serif text-[24px] text-ink dark:text-paper">Mes chantiers</h1>
              <p className="text-sm text-ink-3">
                Bonjour {worker.name || 'compagnon'} — voici tous les sites que vous avez scannés.
              </p>
            </div>
          </div>
          <Link
            href="/worker/scanner"
            className="inline-flex items-center gap-2 border border-orange px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-orange transition hover:bg-paper-2"
          >
            Scanner un nouveau chantier
          </Link>
        </div>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-8">
        <section className="rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3">En cours</p>
              <h2 className="font-serif text-[22px] text-ink dark:text-paper">Chantiers actifs</h2>
            </div>
            <span className="rounded-sm border border-rule-soft font-mono text-[10px] uppercase tracking-widest px-2 py-0.5 text-ink-3">
              {inProgressSites.length} actif(s)
            </span>
          </div>
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            {inProgressSites.length ? (
              inProgressSites.map(({ site, nextTaskDate, tasksCount }) => (
                <div
                  key={site.id}
                  className="flex flex-col gap-3 rounded border border-rule-soft bg-paper-2 p-4 dark:border-rule"
                >
                  <div>
                    <h3 className="font-serif text-[18px] text-ink dark:text-paper">{site.name || 'Chantier'}</h3>
                    <p className="text-xs text-ink-3">
                      {tasksCount} tâche(s) assignée(s)
                    </p>
                  </div>
                  <p className="text-sm text-ink-2">
                    Prochaine mission : {nextTaskDate ? formatDate(nextTaskDate) : 'à définir'}
                  </p>
                </div>
              ))
            ) : (
              <p className="text-sm text-ink-3">
                Aucun chantier actif pour le moment. Scannez un QR code pour en ajouter un nouveau.
              </p>
            )}
          </div>
        </section>

        <section className="rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3">Terminés</p>
              <h2 className="font-serif text-[22px] text-ink dark:text-paper">Chantiers archivés</h2>
            </div>
            <span className="rounded-sm border border-rule-soft font-mono text-[10px] uppercase tracking-widest px-2 py-0.5 text-ink-3">
              {completedSites.length} terminé(s)
            </span>
          </div>
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            {completedSites.length ? (
              completedSites.map(({ site }) => (
                <div
                  key={site.id}
                  className="rounded border border-rule-soft bg-paper-2 p-4 text-sm text-ink-2 dark:border-rule"
                >
                  <p className="font-serif text-[16px] text-ink dark:text-paper">{site.name || 'Chantier'}</p>
                  <p className="text-xs text-ink-3">
                    Terminé le {site.completed_at ? formatDate(site.completed_at) : 'date inconnue'}
                  </p>
                </div>
              ))
            ) : (
              <p className="text-sm text-ink-3">
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
