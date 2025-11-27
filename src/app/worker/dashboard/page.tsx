import Image from 'next/image';
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

  let activeSite: { id: string; name: string | null; deadline: string | null; postal_code: string | null } | null = null;
  let taskPreview:
    | Array<{ id: string; title: string | null; status: string | null; created_at: string | null }>
    | null = null;

  if (worker.site_id) {
    const { data: siteData } = await supabase
      .from('sites')
      .select('id, name, deadline, postal_code')
      .eq('id', worker.site_id)
      .single();
    activeSite = siteData ?? null;

    const { data: tasks } = await supabase
      .from('tasks')
      .select('id, title, status, created_at')
      .eq('site_id', worker.site_id)
      .eq('assigned_worker_id', worker.id)
      .order('created_at', { ascending: true })
      .limit(3);
    taskPreview = tasks ?? null;
  }

  const doneStatuses = new Set(['done', 'completed', 'terminé', 'validated', 'valide', 'réalisé']);
  const totalTasks = taskPreview?.length ?? 0;
  const completedTasks =
    taskPreview?.filter((task) => doneStatuses.has((task.status ?? '').toLowerCase())).length ?? 0;

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-50 via-white to-zinc-100 dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950">
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
          <div className="rounded-3xl border border-zinc-200 bg-white/90 p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900/90">
            <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">Chantier actif</p>
            {activeSite ? (
              <>
                <h2 className="mt-2 text-2xl font-semibold text-zinc-900 dark:text-white">
                  {activeSite.name || 'Chantier en cours'}
                </h2>
                <div className="mt-4 grid gap-4 text-sm text-zinc-600 dark:text-zinc-300 sm:grid-cols-2">
                  <div>
                    <p className="text-xs uppercase tracking-[0.3em] text-zinc-400">Deadline</p>
                    <p className="text-base font-semibold text-zinc-900 dark:text-white">
                      {activeSite.deadline
                        ? new Date(activeSite.deadline).toLocaleDateString('fr-FR', {
                            day: '2-digit',
                            month: 'short',
                          })
                        : 'Non définie'}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.3em] text-zinc-400">Code postal</p>
                    <p className="text-base font-semibold text-zinc-900 dark:text-white">
                      {activeSite.postal_code ?? '—'}
                    </p>
                  </div>
                </div>
                <div className="mt-6 rounded-2xl border border-zinc-200 bg-zinc-50/80 p-4 dark:border-zinc-800 dark:bg-zinc-900/60">
                  <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
                    <span>Tâches assignées</span>
                    <span>
                      {completedTasks}/{totalTasks} terminées
                    </span>
                  </div>
                  <div className="mt-3 space-y-3">
                    {taskPreview && taskPreview.length > 0 ? (
                      taskPreview.map((task) => (
                        <div
                          key={task.id}
                          className="rounded-2xl border border-emerald-200 bg-gradient-to-r from-emerald-50 to-white/70 px-4 py-3 dark:border-emerald-900 dark:from-emerald-950/40 dark:to-zinc-900/40"
                        >
                          <p className="text-sm font-semibold text-emerald-900 dark:text-emerald-100">
                            {task.title || 'Tâche à venir'}
                          </p>
                          <div className="mt-1 flex items-center justify-between text-xs text-emerald-700 dark:text-emerald-200">
                            <span>{formatStatus(task.status)}</span>
                            {task.created_at ? (
                              <span>
                                Ajoutée le{' '}
                                {new Date(task.created_at).toLocaleDateString('fr-FR', {
                                  day: '2-digit',
                                  month: 'short',
                                })}
                              </span>
                            ) : null}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="rounded-2xl border border-dashed border-zinc-300 px-4 py-3 text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
                        Aucune tâche assignée pour le moment. Votre chef de chantier vous enverra bientôt de nouvelles
                        missions.
                      </div>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="mt-4 space-y-3">
                <h2 className="text-2xl font-semibold text-zinc-900 dark:text-white">Pas encore de chantier</h2>
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  Scannez un QR code pour rejoindre un chantier et recevoir instantanément votre planning.
                </p>
                <p className="text-xs text-zinc-500 dark:text-zinc-500">
                  Astuce : gardez votre session ouverte pour accéder à vos tâches même sans connexion.
                </p>
              </div>
            )}
          </div>
          <div>
            <WorkerScanner />
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

