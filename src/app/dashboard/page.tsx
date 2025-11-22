import Link from 'next/link';
import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { CreateSiteForm } from './create-site-form';
import { signOutAction } from '../actions';
import { AppShell } from '@/components/app-shell';
import { DashboardCharts } from './dashboard-charts';
import { SitePlanningMini } from '@/components/site-planning-mini';
import { generatePlanning } from '@/lib/ai/planning';

export const metadata = {
  title: 'Dashboard | ChantiFlow',
};

export default async function DashboardPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: sites, error } = await supabase
    .from('sites')
    .select('id, name, deadline, created_at, completed_at')
    .eq('created_by', user.id)
    .order('created_at', { ascending: false });

  const siteIds = sites?.map((site) => site.id) ?? [];
  const completedSites = (sites ?? []).filter((site) => site.completed_at);
  const activeSites = (sites ?? []).filter((site) => !site.completed_at);

  let totalTasks = 0;
  let doneTasks = 0;

  if (siteIds.length) {
    const { data: tasks } = await supabase
      .from('tasks')
      .select('status')
      .in('site_id', siteIds);

    totalTasks = tasks?.length ?? 0;
    doneTasks = tasks?.filter((task) => task.status === 'done').length ?? 0;
  }

  const pendingTasks = totalTasks - doneTasks;

  const nextDeadlines = (activeSites ?? [])
    .filter((site) => site.deadline)
    .sort((a, b) => (a.deadline ?? '').localeCompare(b.deadline ?? ''))
    .slice(0, 4);

  // Récupérer les workers en attente de validation
  let pendingWorkers: any[] = [];
  try {
    const { data: testData, error: testError } = await supabase
      .from('workers')
      .select('id, name, email, role, created_at, status')
      .eq('created_by', user.id)
      .is('site_id', null)
      .limit(1);
    
    if (!testError) {
      const { data: allAccountWorkers } = await supabase
        .from('workers')
        .select('id, name, email, role, created_at, status')
        .eq('created_by', user.id)
        .is('site_id', null)
        .order('created_at', { ascending: false });
      
      pendingWorkers = (allAccountWorkers ?? []).filter((w: any) => w.status === 'pending');
    }
  } catch (error) {
    // Ignorer les erreurs si la colonne n'existe pas
    console.warn('Erreur récupération workers en attente:', error);
  }

  // Récupérer les plannings de tous les chantiers (limité à 6 pour performance)
  const sitesWithPlanning = await Promise.all(
    (activeSites ?? []).slice(0, 6).map(async (site) => {
      const [{ data: tasks }, { data: workers }] = await Promise.all([
        supabase
          .from('tasks')
          .select('id, title, required_role, duration_hours, status')
          .eq('site_id', site.id),
        supabase
          .from('workers')
          .select('id, name, email, role')
          .eq('site_id', site.id),
      ]);

      let planning: any[] = [];
      if (tasks && tasks.length > 0) {
        try {
          const planningResult = await generatePlanning(
            tasks || [],
            workers || [],
            site.deadline,
          );
          planning = planningResult.orderedTasks.map((p: any) => ({
            taskId: p.taskId,
            taskTitle: tasks.find((t) => t.id === p.taskId)?.title || 'Tâche',
            startDate: p.startDate,
            endDate: p.endDate,
            assignedWorkerId: p.assignedWorkerId,
          }));
        } catch (error) {
          console.error(`Erreur génération planning pour ${site.name}:`, error);
        }
      }

      return {
        site,
        planning,
        workerCount: workers?.length || 0,
        taskCount: tasks?.length || 0,
      };
    }),
  );

  return (
    <AppShell
      heading="Dashboard général"
      subheading="Vue d'ensemble de tous vos chantiers avec graphiques dynamiques"
      userEmail={user.email}
      primarySite={sites?.[0] ?? null}
      actions={
        <form action={signOutAction}>
          <button
            type="submit"
            className="rounded-full border border-zinc-200 px-4 py-2 text-sm font-medium text-zinc-700 transition hover:border-zinc-900 hover:text-zinc-900 dark:border-zinc-700 dark:text-zinc-200 dark:hover:border-white dark:hover:text-white"
          >
            Se déconnecter
          </button>
        </form>
      }
    >
      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Chantiers</p>
          <p className="mt-2 text-3xl font-semibold">{sites?.length ?? 0}</p>
          <p className="text-sm text-zinc-500">
            {completedSites.length} terminé{completedSites.length > 1 ? 's' : ''}
          </p>
        </div>
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Tâches</p>
          <p className="mt-2 text-3xl font-semibold">{pendingTasks}</p>
          <p className="text-sm text-zinc-500">en attente ({doneTasks} terminées)</p>
        </div>
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Progression</p>
          <p className="mt-2 text-3xl font-semibold">
            {totalTasks ? Math.round((doneTasks / totalTasks) * 100) : 0}%
          </p>
          <div className="mt-3 h-2 rounded-full bg-zinc-200 dark:bg-zinc-800">
            <div
              className="h-full rounded-full bg-emerald-500"
              style={{
                width: `${totalTasks ? Math.round((doneTasks / totalTasks) * 100) : 0}%`,
              }}
            />
          </div>
        </div>
      </section>

      {/* Section Demandes en attente de validation */}
      {pendingWorkers.length > 0 && (
        <section className="mt-8 rounded-3xl border border-amber-200 bg-amber-50 p-6 shadow-sm dark:border-amber-800 dark:bg-amber-900/20">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-amber-900 dark:text-amber-300">
                ⏳ Demandes en attente de validation
              </h2>
              <p className="text-sm text-amber-700 dark:text-amber-400">
                {pendingWorkers.length} nouvelle{pendingWorkers.length > 1 ? 's' : ''} personne{pendingWorkers.length > 1 ? 's' : ''} souhaitant rejoindre votre équipe
              </p>
            </div>
            <Link
              href="/team"
              className="text-xs font-semibold text-amber-700 hover:text-amber-900 dark:text-amber-400 dark:hover:text-amber-300"
            >
              Voir tout →
            </Link>
          </div>
          <div className="space-y-3">
            {pendingWorkers.slice(0, 5).map((worker: any) => (
              <div
                key={worker.id}
                className="rounded-2xl border border-amber-300 bg-white p-4 dark:border-amber-700 dark:bg-zinc-900"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                      {worker.name}
                    </p>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      {worker.role ?? 'Rôle non défini'}
                    </p>
                    <p className="text-xs text-zinc-400 dark:text-zinc-500">
                      {worker.email ?? 'Email non communiqué'}
                    </p>
                    <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                      Demandé le {new Date(worker.created_at).toLocaleDateString('fr-FR')}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <form action="/team/actions" method="post" className="inline">
                      <input type="hidden" name="action" value="approve" />
                      <input type="hidden" name="workerId" value={worker.id} />
                      <button
                        type="submit"
                        className="rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-emerald-700 active:scale-95"
                      >
                        Approuver
                      </button>
                    </form>
                    <form action="/team/actions" method="post" className="inline">
                      <input type="hidden" name="action" value="reject" />
                      <input type="hidden" name="workerId" value={worker.id} />
                      <button
                        type="submit"
                        className="rounded-lg bg-rose-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-rose-700 active:scale-95"
                      >
                        Refuser
                      </button>
                    </form>
                  </div>
                </div>
              </div>
            ))}
            {pendingWorkers.length > 5 && (
              <div className="text-center pt-2">
                <Link
                  href="/team"
                  className="text-sm font-semibold text-amber-700 hover:text-amber-900 dark:text-amber-400 dark:hover:text-amber-300"
                >
                  Voir les {pendingWorkers.length - 5} autre{pendingWorkers.length - 5 > 1 ? 's' : ''} demande{pendingWorkers.length - 5 > 1 ? 's' : ''} →
                </Link>
              </div>
            )}
          </div>
        </section>
      )}

      <section className="mt-8 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-6">
          <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Ajouter un chantier
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Déclare un nouveau site avec sa deadline.
            </p>
            <div className="mt-4">
              <CreateSiteForm />
            </div>
          </div>

          {/* Section Chantiers en cours */}
          {activeSites.length > 0 && (
            <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
                    Chantiers en cours
                  </h2>
                  <p className="text-sm text-zinc-500 dark:text-zinc-400">
                    {activeSites.length} chantier{activeSites.length > 1 ? 's' : ''} actif{activeSites.length > 1 ? 's' : ''}
                  </p>
                </div>
              </div>
              <ul className="grid gap-4 md:grid-cols-2">
                {activeSites.map((site) => (
                  <li
                    key={site.id}
                    className="rounded-2xl border border-zinc-100 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900"
                  >
                    <p className="text-xs uppercase tracking-[0.2em] text-zinc-500 dark:text-zinc-400">
                      Deadline
                    </p>
                    <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                      {site.deadline
                        ? new Date(site.deadline).toLocaleDateString('fr-FR')
                        : 'Non définie'}
                    </p>
                    <div className="mt-3 flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
                        {site.name}
                      </h3>
                    </div>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      Créé le{' '}
                      {new Date(site.created_at ?? '').toLocaleDateString('fr-FR')}
                    </p>
                    <Link
                      href={`/site/${site.id}/dashboard`}
                      className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-black dark:text-white"
                    >
                      Ouvrir le chantier →
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="space-y-6">
          {/* Graphique d'occupation des chantiers */}
          <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
                  Occupation des chantiers
                </h2>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  Vue d'ensemble des plannings et de l'occupation
                </p>
              </div>
              <Link
                href="/planning"
                className="text-xs font-semibold text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
              >
                Voir tout →
              </Link>
            </div>
            {sitesWithPlanning.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2">
                {sitesWithPlanning.map(({ site, planning, workerCount, taskCount }) => (
                  <SitePlanningMini
                    key={site.id}
                    site={site}
                    planning={planning}
                    workerCount={workerCount}
                    taskCount={taskCount}
                  />
                ))}
              </div>
            ) : (
              <p className="text-sm text-zinc-500 dark:text-zinc-400 text-center py-4">
                Aucun planning disponible. Générez un planning pour vos chantiers.
              </p>
            )}
          </div>

          {completedSites.length > 0 && (
            <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
                  Chantiers terminés
                </h2>
                <span className="text-xs text-zinc-500 dark:text-zinc-400">
                  {completedSites.length} chantier{completedSites.length > 1 ? 's' : ''}
                </span>
              </div>
              <ul className="space-y-3">
                {completedSites.slice(0, 4).map((site) => (
                  <li
                    key={site.id}
                    className="rounded-2xl border border-zinc-200 bg-zinc-50 px-4 py-3 text-sm dark:border-zinc-700 dark:bg-zinc-900"
                  >
                    <div className="flex items-center justify-between">
                      <p className="font-semibold text-zinc-900 dark:text-white">{site.name}</p>
                      <span className="text-[10px] uppercase tracking-wide text-zinc-500">
                        Terminé
                      </span>
                    </div>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      Clos le{' '}
                      {site.completed_at
                        ? new Date(site.completed_at).toLocaleDateString('fr-FR')
                        : 'date inconnue'}
                    </p>
                    <Link
                      href={`/site/${site.id}/dashboard`}
                      className="mt-2 inline-flex text-xs font-semibold text-emerald-600 hover:text-emerald-800 dark:text-emerald-300"
                    >
                      Voir le chantier →
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <DashboardCharts
            sites={activeSites}
            totalTasks={totalTasks}
            doneTasks={doneTasks}
            pendingTasks={pendingTasks}
            nextDeadlines={nextDeadlines}
          />
          <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Centralisation
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Accès direct aux interfaces terrain.
            </p>
            <div className="mt-4 grid gap-3">
              <Link
                href={sites?.[0] ? `/qr/${sites[0].id}` : '/dashboard'}
                className="rounded-2xl border border-zinc-200 px-4 py-3 text-sm font-semibold transition hover:border-zinc-900 dark:border-zinc-700 dark:hover:border-white"
              >
                🔗 QR employé
              </Link>
              <Link
                href={sites?.[0] ? `/report/${sites[0].id}` : '/dashboard'}
                className="rounded-2xl border border-zinc-200 px-4 py-3 text-sm font-semibold transition hover:border-zinc-900 dark:border-zinc-700 dark:hover:border-white"
              >
                📑 Rapports chef
              </Link>
              <Link
                href="/sites"
                className="rounded-2xl border border-zinc-200 px-4 py-3 text-sm font-semibold transition hover:border-zinc-900 dark:border-zinc-700 dark:hover:border-white"
              >
                🏗️ Tous les chantiers
              </Link>
            </div>
          </div>
        </div>
      </section>
    </AppShell>
  );
}

