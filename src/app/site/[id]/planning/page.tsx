import { notFound, redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { WeeklyCalendar } from '@/components/weekly-calendar';
import { GeneratePlanningButton } from '@/app/site/[id]/generate-planning-button';
import { generatePlanning } from '@/lib/ai/planning';

type Params = {
  params: Promise<{
    id: string;
  }>;
};

function isValidUuid(value: string | undefined) {
  return Boolean(value && /^[0-9a-fA-F-]{36}$/.test(value));
}

export default async function SitePlanningPage({ params }: Params) {
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

  // Récupérer les tâches et workers du chantier
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

  // Générer le planning
  let planning = null;
  if (tasks && tasks.length > 0) {
    try {
      const planningResult = await generatePlanning(
        tasks || [],
        workers || [],
        site.deadline,
      );

      planning = planningResult.orderedTasks.map((task) => ({
        taskId: task.taskId,
        taskTitle: tasks.find((t) => t.id === task.taskId)?.title || 'Tâche',
        order: task.order,
        startDate: task.startDate,
        endDate: task.endDate,
        assignedWorkerId: task.assignedWorkerId,
        priority: task.priority,
      }));
    } catch (error) {
      console.error('Erreur génération planning:', error);
    }
  }

  return (
    <AppShell
      heading={`Planning - ${site.name}`}
      subheading={`Planning du chantier ${site.name}`}
      userEmail={user.email}
      primarySite={{ id: site.id, name: site.name }}
    >
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">
              {site.name}
            </h2>
            {site.deadline && (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Deadline: {new Date(site.deadline).toLocaleDateString('fr-FR')}
              </p>
            )}
          </div>
          <GeneratePlanningButton siteId={site.id} />
        </div>

        {planning && planning.length > 0 ? (
          <WeeklyCalendar
            planning={planning}
            workers={workers?.map((w) => ({
              id: w.id,
              name: w.name,
              email: w.email || '',
              role: w.role,
            })) || []}
          />
        ) : (
          <div className="rounded-2xl border border-zinc-200 bg-white p-8 text-center dark:border-zinc-800 dark:bg-zinc-900">
            <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-4">
              Aucun planning généré pour le moment.
            </p>
            <GeneratePlanningButton siteId={site.id} />
          </div>
        )}
      </div>
    </AppShell>
  );
}

