import { notFound, redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { GeneratePlanningButton } from '@/app/site/[id]/generate-planning-button';
import { EditablePlanningBoard } from '@/components/editable-planning-board';
import { ModernPlanningView } from '@/components/modern-planning-view';

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
      .select(
        'id, title, required_role, duration_hours, status, planned_start, planned_end, planned_worker_id, planned_order',
      )
      .eq('site_id', site.id),
    supabase
      .from('workers')
      .select('id, name, email, role')
      .eq('site_id', site.id),
  ]);

  const persistedPlanning =
    tasks
      ?.filter((task) => task.planned_start && task.planned_end)
      .map((task) => ({
        taskId: task.id,
        taskTitle: task.title,
        order: task.planned_order ?? 0,
        startDate: task.planned_start as string,
        endDate: task.planned_end as string,
        durationHours: task.duration_hours || 8,
        assignedWorkerId: task.planned_worker_id ?? null,
        status: task.status,
        requiredRole: task.required_role,
      }))
      .sort((a, b) => a.order - b.order)
      .map((task, index, arr) => ({
        ...task,
        priority:
          index === 0 ? ('high' as const) : index >= arr.length - 2 ? ('low' as const) : ('medium' as const),
      })) ?? [];

  // Transformer les tâches en phases pour la vue moderne
  const workerMap = new Map(workers?.map((w) => [w.id, w]) ?? []);
  
  // Grouper par required_role ou utiliser le titre comme phase
  const phasesMap = new Map<string, typeof persistedPlanning>();
  persistedPlanning.forEach((task) => {
    const phaseKey = task.requiredRole || task.taskTitle || 'Autre';
    if (!phasesMap.has(phaseKey)) {
      phasesMap.set(phaseKey, []);
    }
    phasesMap.get(phaseKey)!.push(task);
  });

  const modernPhases = Array.from(phasesMap.entries()).map(([phaseName, phaseTasks]) => {
    const normalizeStatus = (s: string | null | undefined) => (s || '').toLowerCase();
    const now = new Date();
    
    const completedTasks = phaseTasks.filter((t) => {
      const status = normalizeStatus(t.status);
      return status === 'done' || status === 'completed' || status === 'terminé' || status === 'validated';
    }).length;
    
    const inProgressTasks = phaseTasks.filter((t) => {
      const status = normalizeStatus(t.status);
      return status === 'in_progress' || status === 'progress' || status === 'en cours';
    }).length;
    
    // Vérifier si une tâche est en retard (non terminée et date de fin passée)
    const delayedTasks = phaseTasks.filter((t) => {
      const status = normalizeStatus(t.status);
      const isCompleted = status === 'done' || status === 'completed' || status === 'terminé' || status === 'validated';
      if (isCompleted) return false;
      
      if (t.endDate) {
        const endDate = new Date(t.endDate);
        return endDate.getTime() < now.getTime();
      }
      return false;
    }).length;
    
    const totalTasks = phaseTasks.length;
    
    let status: 'completed' | 'in_progress' | 'pending' | 'delayed' = 'pending';
    let progress = 0;
    
    if (completedTasks === totalTasks && totalTasks > 0) {
      status = 'completed';
      progress = 100;
    } else if (delayedTasks > 0) {
      status = 'delayed';
      progress = Math.round((completedTasks / totalTasks) * 100);
    } else if (inProgressTasks > 0 || completedTasks > 0) {
      status = 'in_progress';
      progress = Math.round((completedTasks / totalTasks) * 100);
      if (progress === 0 && inProgressTasks > 0) {
        progress = 50; // Au moins 50% si en cours
      }
    }

    // Trouver le worker assigné (prendre le premier en cours ou le premier assigné)
    const inProgressTask = phaseTasks.find((t) => {
      const taskStatus = normalizeStatus(t.status);
      return (taskStatus === 'in_progress' || taskStatus === 'progress' || taskStatus === 'en cours') && t.assignedWorkerId;
    }) || phaseTasks.find((t) => t.assignedWorkerId);
    
    const assignedWorker = inProgressTask?.assignedWorkerId
      ? workerMap.get(inProgressTask.assignedWorkerId)
      : undefined;

    return {
      id: phaseName,
      name: phaseName,
      status,
      progress,
      assignedWorker: assignedWorker
        ? {
            id: assignedWorker.id,
            name: assignedWorker.name || 'Ouvrier',
          }
        : undefined,
    };
  });

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

        {persistedPlanning.length > 0 ? (
          <div className="space-y-6">
            {/* Vue moderne du planning */}
            <ModernPlanningView
              siteName={site.name}
              phases={modernPhases}
              isAheadOfSchedule={
                site.deadline
                  ? (() => {
                      const now = new Date();
                      const deadline = new Date(site.deadline);
                      const daysUntilDeadline = (deadline.getTime() - now.getTime()) / (1000 * 60 * 60 * 24);
                      // En avance si on a plus de 3 jours avant la deadline et que plus de 50% des tâches sont complétées
                      const completedPhases = modernPhases.filter((p) => p.status === 'completed').length;
                      const completionRate = modernPhases.length > 0 ? completedPhases / modernPhases.length : 0;
                      return daysUntilDeadline > 3 && completionRate > 0.5;
                    })()
                  : false
              }
              showAIBadge={true}
            />
            
            {/* Vue éditable détaillée (optionnelle, peut être masquée) */}
            <details className="rounded-2xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
              <summary className="cursor-pointer text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Voir le planning détaillé (édition)
              </summary>
              <div className="mt-4">
                <EditablePlanningBoard
                  siteId={site.id}
                  initialPlanning={persistedPlanning.map(({ status, requiredRole, ...rest }) => rest)}
                  workers={
                    workers?.map((w) => ({
                      id: w.id,
                      name: w.name,
                      role: w.role,
                    })) || []
                  }
                />
              </div>
            </details>
          </div>
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

