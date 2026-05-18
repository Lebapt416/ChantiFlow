import { notFound, redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { DashboardCharts } from '@/app/dashboard/dashboard-charts';
import { SitePlanningMini } from '@/components/site-planning-mini';
import { generateSiteSummary } from '@/lib/ai/summary';
import { WeatherWidget } from '@/components/weather-widget';
import { getUserPlan, canAccessWeather, canAccessProFeatures } from '@/lib/plans';
import { PDFButton } from '@/components/pdf-button';

type Params = {
  params: Promise<{
    id: string;
  }>;
};

function isValidUuid(value: string | undefined) {
  return Boolean(value && /^[0-9a-fA-F-]{36}$/.test(value));
}

export default async function SiteDashboardPage({ params }: Params) {
  const { id } = await params;

  if (!isValidUuid(id)) {
    notFound();
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: site, error: siteError } = await supabase
    .from('sites')
    .select('*')
    .eq('id', id)
    .single();

  if (siteError || !site || site.created_by !== user.id) {
    notFound();
  }

  // Récupérer les données du chantier
  const [{ data: tasks }, { data: workers }] = await Promise.all([
    supabase
      .from('tasks')
      .select(
        'id, title, required_role, duration_hours, status, planned_start, planned_end, planned_worker_id, planned_order',
      )
      .eq('site_id', site.id),
    supabase
      .from('workers')
      .select('id, name, email, role')
      .eq('site_id', site.id),
  ]);

  const totalTasks = tasks?.length ?? 0;
  const doneTasks = tasks?.filter((task) => task.status === 'done').length ?? 0;
  const pendingTasks = totalTasks - doneTasks;
  const progress = totalTasks > 0 ? Math.round((doneTasks / totalTasks) * 100) : 0;

  const persistedPlanning =
    tasks
      ?.filter((task) => task.planned_start && task.planned_end)
      .map((task) => ({
        taskId: task.id,
        taskTitle: task.title,
        startDate: task.planned_start as string,
        endDate: task.planned_end as string,
        assignedWorkerId: task.planned_worker_id,
        durationHours: task.duration_hours || 8,
        order: task.planned_order ?? 0,
      }))
      .sort((a, b) => a.order - b.order) ?? [];

  const todayKey = new Date().toISOString().split('T')[0];
  const workerMap = new Map(workers?.map((w) => [w.id, w]) ?? []);
  const todaysTasks = persistedPlanning
    .filter((task) => new Date(task.startDate).toISOString().split('T')[0] === todayKey)
    .sort(
      (a, b) => new Date(a.startDate).getTime() - new Date(b.startDate).getTime(),
    );

  // Récupérer le plan de l'utilisateur pour vérifier les accès
  const userPlan = await getUserPlan(user);
  const hasWeatherAccess = canAccessWeather(userPlan);
  const hasProAccess = canAccessProFeatures(userPlan);

  // Générer le résumé IA du chantier (avec météo si accès Plus/Pro)
  let aiSummary: { summary: string; status: string; sites_mentioned?: string[] } | null = null;
  if (tasks && tasks.length > 0) {
    try {
      console.log('🔍 Génération résumé IA pour:', {
        siteId: site.id,
        siteName: site.name,
        tasksCount: tasks.length,
        hasWeatherAccess,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        postalCode: (site as any).postal_code || 'non défini'
      });
      aiSummary = await generateSiteSummary(site, tasks, hasWeatherAccess);
      if (!aiSummary) {
        console.warn('⚠️ Résumé IA retourné null - vérifier les logs API');
      }
    } catch (error) {
      console.error('❌ Erreur génération résumé IA:', error);
      if (error instanceof Error) {
        console.error('Détails erreur:', error.message, error.stack);
      }
    }
  } else {
    console.log('ℹ️ Pas de tâches, résumé IA ignoré');
  }

  return (
    <AppShell
      heading={`Dashboard - ${site.name}`}
      subheading="Vue d'ensemble du chantier"
      userEmail={user.email}
      primarySite={{ id: site.id, name: site.name }}
    >
      <div className="space-y-6">
        {/* Résumé IA du chantier */}
        {aiSummary && (
          <section
            className={`rounded border p-6 ${
              aiSummary.status === 'critical'
                ? 'border-rose-300 bg-gradient-to-br from-rose-50 to-rose-100/50 dark:border-rose-700 dark:from-rose-900/30 dark:to-rose-800/20'
                : aiSummary.status === 'warning'
                  ? 'border-amber-300 bg-gradient-to-br from-amber-50 to-amber-100/50 dark:border-amber-700 dark:from-amber-900/30 dark:to-amber-800/20'
                  : 'border-orange bg-gradient-to-br from-orange to-orange/50 dark:border-orange dark:from-paper-2 dark:to-orange/20'
            }`}
          >
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0">
                {aiSummary.status === 'critical' ? (
                  <div className="rounded-full bg-rose-200 p-3 shadow-md dark:bg-rose-800/50">
                    <span className="text-3xl">⚠️</span>
                  </div>
                ) : aiSummary.status === 'warning' ? (
                  <div className="rounded-full bg-amber-200 p-3 shadow-md dark:bg-amber-800/50">
                    <span className="text-3xl">🟠</span>
                  </div>
                ) : (
                  <div className="rounded-full bg-orange p-3 shadow-md dark:bg-orange/50">
                    <span className="text-3xl">✨</span>
                  </div>
                )}
              </div>
              <div className="flex-1">
                <div className="mb-2 flex items-center gap-2">
                  <h2 className="text-xl font-bold text-zinc-900 dark:text-white">
                    Analyse IA du chantier
                  </h2>
                  <span className="rounded-full bg-paper px-2 py-0.5 text-xs font-semibold text-zinc-700 dark:bg-zinc-800/80 dark:text-zinc-300">
                    IA
                  </span>
                </div>
                <p
                  className={`text-base leading-relaxed ${
                    aiSummary.status === 'critical'
                      ? 'text-rose-900 dark:text-rose-100'
                      : aiSummary.status === 'warning'
                        ? 'text-amber-900 dark:text-amber-100'
                        : 'text-ink dark:text-paper'
                  }`}
                >
                  {aiSummary.summary}
                </p>
              </div>
            </div>
          </section>
        )}

        {/* Stats du chantier */}
        <section className="grid gap-4 md:grid-cols-4">
          <div className="rounded border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
            <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Tâches</p>
            <p className="mt-2 text-3xl font-semibold">{totalTasks}</p>
            <p className="text-sm text-zinc-500">total</p>
          </div>
          <div className="rounded border border-amber-200 bg-amber-50 p-5 dark:border-amber-500/30 dark:bg-amber-900/20">
            <p className="text-xs uppercase tracking-[0.3em] text-amber-800 dark:text-amber-200">
              À traiter
            </p>
            <p className="mt-2 text-3xl font-semibold text-amber-900 dark:text-amber-100">
              {pendingTasks}
            </p>
            <p className="text-sm text-amber-800/80 dark:text-amber-200">en attente</p>
          </div>
          <div className="rounded border border-rule-soft bg-paper-2 p-5 dark:border-orange/30 dark:bg-paper-2">
            <p className="text-xs uppercase tracking-[0.3em] text-ink dark:text-paper">
              Terminées
            </p>
            <p className="mt-2 text-3xl font-semibold text-ink dark:text-orange">
              {doneTasks}
            </p>
            <p className="text-sm text-ink/80 dark:text-paper">
              {progress}% complété
            </p>
          </div>
          <div className="rounded border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
            <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Équipe</p>
            <p className="mt-2 text-3xl font-semibold">{workers?.length ?? 0}</p>
            <p className="text-sm text-zinc-500">membres</p>
          </div>
        </section>

        {/* Planning du jour */}
        <section className="rounded border border-zinc-100 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
                Planning du jour
              </h3>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                {new Date().toLocaleDateString('fr-FR', {
                  weekday: 'long',
                  day: 'numeric',
                  month: 'long',
                })}
              </p>
            </div>
          </div>

          {todaysTasks.length > 0 ? (
            <div className="space-y-3">
              {todaysTasks.map((task) => {
                const worker = task.assignedWorkerId
                  ? workerMap.get(task.assignedWorkerId)
                  : null;
                return (
                  <div
                    key={task.taskId}
                    className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800"
                  >
                    <div className="flex items-center justify-between text-sm">
                      <div>
                        <p className="font-semibold text-zinc-900 dark:text-white">
                          {task.taskTitle}
                        </p>
                        <p className="text-xs text-zinc-500 dark:text-zinc-400">
                          {new Date(task.startDate).toLocaleTimeString('fr-FR', {
                            hour: '2-digit',
                            minute: '2-digit',
                          })}{' '}
                          -{' '}
                          {new Date(task.endDate).toLocaleTimeString('fr-FR', {
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </p>
                      </div>
                      <span className="rounded-full bg-zinc-900 px-3 py-1 text-[11px] font-semibold text-white dark:bg-zinc-100 dark:text-zinc-900">
                        {task.durationHours}h
                      </span>
                    </div>
                    <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
                      {worker ? `${worker.name} • ${worker.role ?? 'Rôle non défini'}` : 'Non assigné'}
                    </p>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Aucune tâche planifiée pour aujourd&apos;hui. Glissez-déposez une tâche sur la journée
              souhaitée dans l&apos;onglet Planning pour la programmer.
            </p>
          )}
        </section>

        {/* Widget Météo */}
        <WeatherWidget location={site.postal_code || undefined} isLocked={!hasWeatherAccess} />

        {/* Section Rapports avec bouton PDF */}
        <section className="rounded border border-zinc-100 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">Rapports</h3>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Exportez vos données de chantier
              </p>
            </div>
            <PDFButton siteId={site.id} isPro={hasProAccess} />
          </div>
        </section>

        {/* Planning du chantier */}
        <SitePlanningMini
          site={site}
          planning={persistedPlanning}
          workerCount={workers?.length || 0}
          taskCount={tasks?.length || 0}
        />

        {/* Graphiques */}
        <DashboardCharts
          sites={[site]}
          totalTasks={totalTasks}
          doneTasks={doneTasks}
          pendingTasks={pendingTasks}
          nextDeadlines={site.deadline ? [site] : []}
        />
      </div>
    </AppShell>
  );
}

