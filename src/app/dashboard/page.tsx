import Link from 'next/link';
import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { CreateSiteForm } from './create-site-form';
import { signOutAction } from '../actions';
import { AppShell } from '@/components/app-shell';
import { DashboardCharts } from './dashboard-charts';
import { SitePlanningMini } from '@/components/site-planning-mini';
import { generatePlanning } from '@/lib/ai/planning';
import { AIStatusBadge } from '@/components/ai-status-badge';
import { generateGlobalSummary } from '@/lib/ai/summary';

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

  const { data: sites } = await supabase
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let pendingWorkers: any[] = [];
  try {
    const { error: testError } = await supabase
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

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      pendingWorkers = (allAccountWorkers ?? []).filter((w: any) => w.status === 'pending');
    }
  } catch (error) {
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

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let planning: any[] = [];
      if (tasks && tasks.length > 0) {
        try {
          const planningResult = await generatePlanning(
            tasks || [],
            workers || [],
            site.deadline,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (site as any).postal_code || undefined,
          );
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
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

  // Générer le résumé IA global
  let aiSummary: { summary: string; status: string; sites_mentioned?: string[] } | null = null;
  if (activeSites.length > 0) {
    try {
      const { data: allTasks } = await supabase
        .from('tasks')
        .select('id, status, required_role, duration_hours, site_id')
        .in('site_id', activeSites.map((s) => s.id));

      const sitesData = activeSites.map((site) => {
        const siteTasks = allTasks?.filter((t) => t.site_id === site.id) || [];
        const doneTasks = siteTasks.filter((t) => t.status === 'done');

        const roleDiversity = new Set(
          siteTasks.map((t) => t.required_role).filter(Boolean),
        ).size;
        const avgDuration =
          siteTasks.reduce((sum, t) => sum + (t.duration_hours || 8), 0) /
          Math.max(1, siteTasks.length);
        const complexity = Math.min(10, Math.max(1, avgDuration / 4 + roleDiversity / 2));

        const daysElapsed = Math.ceil(
          (new Date().getTime() - new Date(site.created_at).getTime()) /
            (1000 * 3600 * 24),
        );

        return {
          ...site,
          tasks: siteTasks,
          tasks_total: siteTasks.length,
          tasks_done: doneTasks.length,
          complexity: Number(complexity.toFixed(2)),
          days_elapsed: daysElapsed,
        };
      });

      aiSummary = await generateGlobalSummary(sitesData);
    } catch (error) {
      console.error('Erreur génération résumé IA:', error);
    }
  }

  return (
    <AppShell
      heading="Dashboard général"
      subheading="Vue d'ensemble de tous vos chantiers avec graphiques dynamiques"
      userEmail={user.email}
      primarySite={sites?.[0] ?? null}
      actions={
        <div className="flex items-center gap-3">
          <AIStatusBadge />
          <form action={signOutAction}>
            <button
              type="submit"
              className="border border-rule-soft px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-ink-2 transition hover:border-rule hover:text-ink dark:border-rule dark:text-ink-2"
            >
              Se déconnecter
            </button>
          </form>
        </div>
      }
    >
      {/* Résumé IA Global */}
      {aiSummary && (
        <section
          className={`mb-8 rounded border p-6 ${
            aiSummary.status === 'critical'
              ? 'border-danger bg-paper-2'
              : aiSummary.status === 'warning'
                ? 'border-warn bg-paper-2'
                : 'border-orange bg-paper-2'
          }`}
        >
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0">
              {aiSummary.status === 'critical' ? (
                <div className="bg-paper-2 p-3">
                  <span className="text-3xl">⚠️</span>
                </div>
              ) : aiSummary.status === 'warning' ? (
                <div className="bg-paper-2 p-3">
                  <span className="text-3xl">🟠</span>
                </div>
              ) : (
                <div className="bg-paper-2 p-3">
                  <span className="text-3xl">✨</span>
                </div>
              )}
            </div>
            <div className="flex-1">
              <div className="mb-2 flex items-center gap-2">
                <h2 className="font-serif text-[22px] text-ink dark:text-paper">
                  Analyse IA des chantiers
                </h2>
                <span className="rounded-sm border border-rule-soft font-mono text-[10px] uppercase tracking-widest px-2 py-0.5 text-ink-3">
                  IA
                </span>
              </div>
              <p
                className={`text-base leading-relaxed ${
                  aiSummary.status === 'critical'
                    ? 'text-danger'
                    : aiSummary.status === 'warning'
                      ? 'text-warn'
                      : 'text-ink dark:text-paper'
                }`}
              >
                {aiSummary.summary}
              </p>
              {aiSummary.sites_mentioned &&
               aiSummary.sites_mentioned.length > 0 &&
               (aiSummary.status === 'critical' || aiSummary.status === 'warning') && (
                <div className="mt-4 flex flex-wrap gap-2">
                  {aiSummary.sites_mentioned.map((siteName: string, index: number) => {
                    const site = activeSites.find((s) => s.name === siteName);
                    if (!site) return null;
                    return (
                      <Link
                        key={index}
                        href={`/site/${site.id}/dashboard`}
                        className={`inline-flex items-center gap-1.5 border px-3 py-1.5 font-mono text-[10px] uppercase tracking-widest transition ${
                          aiSummary.status === 'critical'
                            ? 'border-danger text-danger hover:bg-paper-2'
                            : 'border-warn text-warn hover:bg-paper-2'
                        }`}
                      >
                        <span>🏗️</span>
                        <span>{siteName}</span>
                        <span>→</span>
                      </Link>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded border border-rule-soft bg-paper p-5 dark:border-rule dark:bg-ink">
          <p className="text-xs uppercase tracking-[0.3em] text-ink-3">Chantiers</p>
          <p className="mt-2 text-3xl font-semibold text-ink dark:text-paper">{sites?.length ?? 0}</p>
          <p className="text-sm text-ink-3">
            {completedSites.length} terminé{completedSites.length > 1 ? 's' : ''}
          </p>
        </div>
        <div className="rounded border border-rule-soft bg-paper p-5 dark:border-rule dark:bg-ink">
          <p className="text-xs uppercase tracking-[0.3em] text-ink-3">Tâches</p>
          <p className="mt-2 text-3xl font-semibold text-ink dark:text-paper">{pendingTasks}</p>
          <p className="text-sm text-ink-3">en attente ({doneTasks} terminées)</p>
        </div>
        <div className="rounded border border-rule-soft bg-paper p-5 dark:border-rule dark:bg-ink">
          <p className="text-xs uppercase tracking-[0.3em] text-ink-3">Progression</p>
          <p className="mt-2 text-3xl font-semibold text-ink dark:text-paper">
            {totalTasks ? Math.round((doneTasks / totalTasks) * 100) : 0}%
          </p>
          <div className="mt-3 h-0.5 bg-paper-2 border border-rule-soft/50">
            <div
              className="h-full bg-orange"
              style={{
                width: `${totalTasks ? Math.round((doneTasks / totalTasks) * 100) : 0}%`,
              }}
            />
          </div>
        </div>
      </section>

      {/* Section Demandes en attente de validation */}
      {pendingWorkers.length > 0 && (
        <section className="mt-8 rounded border border-warn bg-paper-2 p-6">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h2 className="font-serif text-[22px] text-warn">
                ⏳ Demandes en attente de validation
              </h2>
              <p className="text-sm text-ink-2">
                {pendingWorkers.length} nouvelle{pendingWorkers.length > 1 ? 's' : ''} personne{pendingWorkers.length > 1 ? 's' : ''} souhaitant rejoindre votre équipe
              </p>
            </div>
            <Link
              href="/team"
              className="font-mono text-[10px] uppercase tracking-widest text-warn hover:text-ink"
            >
              Voir tout →
            </Link>
          </div>
          <div className="space-y-3">
            {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
            {pendingWorkers.slice(0, 5).map((worker: any) => (
              <div
                key={worker.id}
                className="rounded border border-warn bg-paper p-4"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-ink dark:text-paper">
                      {worker.name}
                    </p>
                    <p className="text-xs text-ink-3">
                      {worker.role ?? 'Rôle non défini'}
                    </p>
                    <p className="text-xs text-ink-3">
                      {worker.email ?? 'Email non communiqué'}
                    </p>
                    <p className="font-mono text-[10px] text-warn mt-1">
                      Demandé le {new Date(worker.created_at).toLocaleDateString('fr-FR')}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <form action="/team/actions" method="post" className="inline">
                      <input type="hidden" name="action" value="approve" />
                      <input type="hidden" name="workerId" value={worker.id} />
                      <button
                        type="submit"
                        className="border border-orange px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-orange transition hover:bg-paper-2"
                      >
                        Approuver
                      </button>
                    </form>
                    <form action="/team/actions" method="post" className="inline">
                      <input type="hidden" name="action" value="reject" />
                      <input type="hidden" name="workerId" value={worker.id} />
                      <button
                        type="submit"
                        className="border border-danger px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-danger transition hover:bg-paper-2"
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
                  className="font-mono text-[10px] uppercase tracking-widest text-warn hover:text-ink"
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
          <div className="rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
            <h2 className="font-serif text-[22px] text-ink dark:text-paper">
              Ajouter un chantier
            </h2>
            <p className="text-sm text-ink-3">
              Déclare un nouveau site avec sa deadline.
            </p>
            <div className="mt-4">
              <CreateSiteForm />
            </div>
          </div>

          {/* Section Chantiers en cours */}
          {activeSites.length > 0 && (
            <div className="rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="font-serif text-[22px] text-ink dark:text-paper">
                    Chantiers en cours
                  </h2>
                  <p className="text-sm text-ink-3">
                    {activeSites.length} chantier{activeSites.length > 1 ? 's' : ''} actif{activeSites.length > 1 ? 's' : ''}
                  </p>
                </div>
              </div>
              <ul className="grid gap-4 md:grid-cols-2">
                {activeSites.map((site) => (
                  <li
                    key={site.id}
                    className="rounded border border-rule-soft bg-paper-2 p-4 dark:border-rule"
                  >
                    <p className="text-xs uppercase tracking-[0.2em] text-ink-3">
                      Deadline
                    </p>
                    <p className="text-sm font-semibold text-ink dark:text-paper">
                      {site.deadline
                        ? new Date(site.deadline).toLocaleDateString('fr-FR')
                        : 'Non définie'}
                    </p>
                    <div className="mt-3 flex items-center justify-between">
                      <h3 className="font-serif text-[18px] text-ink dark:text-paper">
                        {site.name}
                      </h3>
                    </div>
                    <p className="text-xs text-ink-3">
                      Créé le{' '}
                      {new Date(site.created_at ?? '').toLocaleDateString('fr-FR')}
                    </p>
                    <Link
                      href={`/site/${site.id}/dashboard`}
                      className="mt-4 inline-flex items-center gap-2 font-mono text-[10px] uppercase tracking-widest text-ink-2 hover:text-ink"
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
          <div className="rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h2 className="font-serif text-[22px] text-ink dark:text-paper">
                  Occupation des chantiers
                </h2>
                <p className="text-sm text-ink-3">
                  Vue d&apos;ensemble des plannings et de l&apos;occupation
                </p>
              </div>
              <Link
                href="/planning"
                className="font-mono text-[10px] uppercase tracking-widest text-ink-2 hover:text-ink"
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
              <p className="text-sm text-ink-3 text-center py-4">
                Aucun planning disponible. Générez un planning pour vos chantiers.
              </p>
            )}
          </div>

          {completedSites.length > 0 && (
            <div className="rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="font-serif text-[22px] text-ink dark:text-paper">
                  Chantiers terminés
                </h2>
                <span className="font-mono text-[10px] uppercase tracking-widest text-ink-3">
                  {completedSites.length} chantier{completedSites.length > 1 ? 's' : ''}
                </span>
              </div>
              <ul className="space-y-3">
                {completedSites.slice(0, 4).map((site) => (
                  <li
                    key={site.id}
                    className="rounded border border-rule-soft bg-paper-2 px-4 py-3 text-sm dark:border-rule"
                  >
                    <div className="flex items-center justify-between">
                      <p className="font-semibold text-ink dark:text-paper">{site.name}</p>
                      <span className="font-mono text-[10px] uppercase tracking-widest text-ink-3">
                        Terminé
                      </span>
                    </div>
                    <p className="text-xs text-ink-3">
                      Clos le{' '}
                      {site.completed_at
                        ? new Date(site.completed_at).toLocaleDateString('fr-FR')
                        : 'date inconnue'}
                    </p>
                    <Link
                      href={`/site/${site.id}/dashboard`}
                      className="mt-2 inline-flex font-mono text-[10px] uppercase tracking-widest text-orange hover:text-ink dark:text-green"
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
          <div className="rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
            <h2 className="font-serif text-[22px] text-ink dark:text-paper">
              Centralisation
            </h2>
            <p className="text-sm text-ink-3">
              Accès direct aux interfaces terrain.
            </p>
            <div className="mt-4 grid gap-3">
              <Link
                href={sites?.[0] ? `/qr/${sites[0].id}` : '/dashboard'}
                className="rounded border border-rule-soft px-4 py-3 text-sm font-semibold text-ink transition hover:border-rule dark:border-rule"
              >
                🔗 QR employé
              </Link>
              <Link
                href={sites?.[0] ? `/report/${sites[0].id}` : '/dashboard'}
                className="rounded border border-rule-soft px-4 py-3 text-sm font-semibold text-ink transition hover:border-rule dark:border-rule"
              >
                📑 Rapports chef
              </Link>
              <Link
                href="/sites"
                className="rounded border border-rule-soft px-4 py-3 text-sm font-semibold text-ink transition hover:border-rule dark:border-rule"
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
