import Image from 'next/image';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { readWorkerSession } from '@/lib/worker-session';
import { WorkerSiteShell } from '../components/worker-site-shell';
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

  let sitesById = new Map<string, { id: string; name: string | null; deadline: string | null; postal_code: string | null }>();
  if (siteIds.size > 0) {
    const { data: sitesData } = await supabase
      .from('sites')
      .select('id, name, deadline, postal_code')
      .in('id', Array.from(siteIds));
    sitesById = new Map((sitesData ?? []).map((site) => [site.id, site]));
  }

  const activeSite = worker.site_id ? sitesById.get(worker.site_id) ?? null : null;

  const doneStatuses = new Set(['done', 'completed', 'terminé', 'validated', 'valide', 'réalisé']);

  const upcomingTimeline =
    assignedTasks
      ?.filter((task) => !worker.site_id || task.site_id === worker.site_id)
      .map((task) => ({
        id: task.id,
        title: task.title,
        date: task.planned_start || task.planned_end || task.created_at,
        siteId: task.site_id,
        status: task.status,
      }))
      .filter((task) => task.date)
      .sort((a, b) => new Date(a.date ?? 0).getTime() - new Date(b.date ?? 0).getTime())
      .slice(0, 6) ?? [];

  const totalTasks = assignedTasks?.length ?? 0;
  const completedTasks =
    assignedTasks?.filter((task) => doneStatuses.has((task.status ?? '').toLowerCase())).length ?? 0;

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
              <h1 className="text-2xl font-semibold text-zinc-900 dark:text-white">Espace ouvrier</h1>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Bonjour {worker.name || 'compagnon'} — votre session reste ouverte même si vous fermez l&apos;app.
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 text-xs font-semibold">
            <span className="rounded-full bg-emerald-50 px-4 py-1 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200">
              ✅ Session persistante
            </span>
            <span className="rounded-full bg-zinc-100 px-4 py-1 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-200">
              {activeSite ? `Chantier · ${activeSite.name}` : 'Pas de chantier actif'}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-8">
        <section className="grid gap-6 lg:grid-cols-2">
          <div className="lg:col-span-2 rounded-3xl border border-zinc-200 bg-white/90 p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900/90">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">Vue d&apos;ensemble</p>
                <h2 className="text-2xl font-semibold text-zinc-900 dark:text-white">
                  {activeSite ? activeSite.name || 'Votre chantier' : 'Prêt pour votre prochain chantier'}
                </h2>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  {activeSite
                    ? 'Voici vos prochaines missions. Vous pouvez en ajouter d’autres en scannant un QR code.'
                    : 'Rejoignez un chantier pour recevoir automatiquement votre planning.'}
                </p>
              </div>
              <div className="rounded-2xl border border-zinc-200 bg-zinc-50 px-4 py-2 text-right text-sm font-semibold text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900/70 dark:text-zinc-200">
                <p>Tâches</p>
                <p className="text-lg text-zinc-900 dark:text-white">
                  {completedTasks}/{totalTasks} terminées
                </p>
              </div>
            </div>
            <div className="mt-6 grid gap-4 text-sm text-zinc-600 dark:text-zinc-300 sm:grid-cols-2">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-zinc-400">Deadline</p>
                <p className="text-base font-semibold text-zinc-900 dark:text-white">
                  {activeSite?.deadline ? formatDate(activeSite.deadline) : 'Non définie'}
                </p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-zinc-400">Code postal</p>
                <p className="text-base font-semibold text-zinc-900 dark:text-white">
                  {activeSite?.postal_code ?? '—'}
                </p>
              </div>
            </div>
            <div className="mt-6 rounded-2xl border border-zinc-200 bg-zinc-50/80 p-4 dark:border-zinc-800 dark:bg-zinc-900/60">
              <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
                <span>Prochaines missions</span>
                <span className="text-zinc-900 dark:text-white">
                  {upcomingTimeline.length} à venir
                </span>
              </div>
              <div className="mt-3 space-y-3">
                {upcomingTimeline.length ? (
                  upcomingTimeline.map((task) => {
                    const site = task.siteId ? sitesById.get(task.siteId) : null;
                    return (
                      <div
                        key={task.id}
                        className="rounded-2xl border border-emerald-200 bg-gradient-to-r from-emerald-50 to-white px-4 py-3 dark:border-emerald-900 dark:from-emerald-950/40 dark:to-zinc-900/40"
                      >
                        <div className="flex items-center justify-between text-xs text-emerald-700 dark:text-emerald-200">
                          <span>{formatDate(task.date)}</span>
                          <span>{site?.name || 'Chantier'}</span>
                        </div>
                        <p className="mt-1 text-sm font-semibold text-emerald-900 dark:text-emerald-100">
                          {task.title || 'Tâche à venir'}
                        </p>
                        <p className="text-xs text-emerald-700 dark:text-emerald-200">
                          {formatStatus(task.status)}
                        </p>
                      </div>
                    );
                  })
                ) : (
                  <div className="rounded-2xl border border-dashed border-zinc-300 px-4 py-3 text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
                    Aucune mission planifiée pour l’instant. Scannez un chantier pour démarrer.
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="rounded-3xl border border-zinc-200 bg-white/90 p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900/90">
            <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">Ajouter un chantier</p>
            <h2 className="mt-2 text-xl font-semibold text-zinc-900 dark:text-white">Scanner un QR code</h2>
            <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
              Besoin de rejoindre un nouveau chantier ? Ouvre le scanner dédié pour te connecter instantanément.
            </p>
            <div className="mt-6 flex flex-col gap-3">
              <Link
                href="/worker/scanner"
                className="inline-flex items-center justify-center gap-2 rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-emerald-700 active:scale-[0.99]"
              >
                Ouvrir le scanner
              </Link>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Astuce : une fois la caméra autorisée, l&apos;accès reste ouvert pour les prochains scans.
              </p>
            </div>
          </div>
        </section>

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
      <WorkerNav />
    </div>
  );
}

function formatStatus(status?: string | null) {
  if (!status) return 'En attente';
  const normalized = status.toLowerCase();
  if (['done', 'completed', 'terminé', 'validated', 'valide', 'réalisé'].includes(normalized)) {
    return 'Terminé';
  }
  if (['in_progress', 'progress', 'en cours'].includes(normalized)) {
    return 'En cours';
  }
  if (['blocked', 'blocked_alert', 'bloqué'].includes(normalized)) {
    return 'Bloqué';
  }
  return status.charAt(0).toUpperCase() + status.slice(1);
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

