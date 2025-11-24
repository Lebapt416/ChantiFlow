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

  // Générer le résumé IA du chantier
  let aiSummary: { summary: string; status: string } | null = null;
  if (tasks && tasks.length > 0) {
    try {
      aiSummary = await generateSiteSummary(site, tasks);
    } catch (error) {
      console.error('Erreur génération résumé IA:', error);
    }
  }

  // Récupérer le plan de l'utilisateur pour vérifier les accès
  const userPlan = await getUserPlan(user);
  const hasWeatherAccess = canAccessWeather(userPlan);
  const hasProAccess = canAccessProFeatures(userPlan);

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
            className={`rounded-3xl border p-6 shadow-sm ${
              aiSummary.status === 'critical'
                ? 'border-rose-200 bg-rose-50 dark:border-rose-800 dark:bg-rose-900/20'
                : aiSummary.status === 'warning'
                  ? 'border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-900/20'
                  : 'border-emerald-200 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-900/20'
            }`}
          >
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0">
                {aiSummary.status === 'critical' ? (
                  <div className="rounded-full bg-rose-100 p-2 dark:bg-rose-900/30">
                    <span className="text-2xl">⚠️</span>
                  </div>
                ) : aiSummary.status === 'warning' ? (
                  <div className="rounded-full bg-amber-100 p-2 dark:bg-amber-900/30">
                    <span className="text-2xl">🟠</span>
                  </div>
                ) : (
                  <div className="rounded-full bg-emerald-100 p-2 dark:bg-emerald-900/30">
                    <span className="text-2xl">✨</span>
                  </div>
                )}
              </div>
              <div className="flex-1">
                <h2 className="mb-1 text-lg font-semibold text-zinc-900 dark:text-white">
                  Analyse IA du chantier
                </h2>
                <p
                  className={`text-sm ${
                    aiSummary.status === 'critical'
                      ? 'text-rose-900 dark:text-rose-200'
                      : aiSummary.status === 'warning'
                        ? 'text-amber-900 dark:text-amber-200'
                        : 'text-emerald-900 dark:text-emerald-200'
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
          <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
            <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Tâches</p>
            <p className="mt-2 text-3xl font-semibold">{totalTasks}</p>
            <p className="text-sm text-zinc-500">total</p>
          </div>
          <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5 dark:border-amber-500/30 dark:bg-amber-900/20">
            <p className="text-xs uppercase tracking-[0.3em] text-amber-800 dark:text-amber-200">
              À traiter
            </p>
            <p className="mt-2 text-3xl font-semibold text-amber-900 dark:text-amber-100">
              {pendingTasks}
            </p>
            <p className="text-sm text-amber-800/80 dark:text-amber-200">en attente</p>
          </div>
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-5 dark:border-emerald-500/30 dark:bg-emerald-900/20">
            <p className="text-xs uppercase tracking-[0.3em] text-emerald-800 dark:text-emerald-100">
              Terminées
            </p>
            <p className="mt-2 text-3xl font-semibold text-emerald-900 dark:text-emerald-50">
              {doneTasks}
            </p>
            <p className="text-sm text-emerald-800/80 dark:text-emerald-100">
              {progress}% complété
            </p>
          </div>
          <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
            <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Équipe</p>
            <p className="mt-2 text-3xl font-semibold">{workers?.length ?? 0}</p>
            <p className="text-sm text-zinc-500">membres</p>
          </div>
        </section>

        {/* Planning du jour */}
        <section className="rounded-2xl border border-zinc-100 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
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
              Aucune tâche planifiée pour aujourd'hui. Glissez-déposez une tâche sur la journée
              souhaitée dans l'onglet Planning pour la programmer.
            </p>
          )}
        </section>

        {/* Widget Météo */}
        <WeatherWidget location={site.address || undefined} isLocked={!hasWeatherAccess} />

        {/* Section Rapports avec bouton PDF */}
        <section className="rounded-2xl border border-zinc-100 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
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

