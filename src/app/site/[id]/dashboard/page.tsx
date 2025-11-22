import { notFound, redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { DashboardCharts } from '@/app/dashboard/dashboard-charts';
import { SitePlanningMini } from '@/components/site-planning-mini';
import { generatePlanning } from '@/lib/ai/planning';

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
      .select('id, title, required_role, duration_hours, status')
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

  // Générer le planning pour ce chantier
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

  return (
    <AppShell
      heading={`Dashboard - ${site.name}`}
      subheading="Vue d'ensemble du chantier"
      userEmail={user.email}
      primarySite={{ id: site.id, name: site.name }}
    >
      <div className="space-y-6">
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

        {/* Planning du chantier */}
        <SitePlanningMini
          site={site}
          planning={planning}
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

