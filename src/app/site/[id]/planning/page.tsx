import { notFound, redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { GeneratePlanningButton } from '@/app/site/[id]/generate-planning-button';
import { PlanningViewEditorial } from '@/components/planning-view-editorial';

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

  const planningTasks = (tasks ?? [])
    .filter((t) => t.planned_start && t.planned_end)
    .sort((a, b) => (a.planned_order ?? 0) - (b.planned_order ?? 0))
    .map((t, i) => ({
      taskId: t.id,
      taskTitle: t.title,
      startDate: t.planned_start as string,
      endDate: t.planned_end as string,
      status: t.status,
      requiredRole: t.required_role,
      assignedWorkerId: t.planned_worker_id ?? null,
      durationHours: t.duration_hours || 8,
      order: t.planned_order ?? i,
      priority: (i === 0 ? 'high' : i >= (tasks?.length ?? 0) - 2 ? 'low' : 'medium') as 'high' | 'medium' | 'low',
    }));

  const planningWorkers = (workers ?? []).map((w) => ({
    id: w.id,
    name: w.name,
    role: w.role,
  }));

  return (
    <AppShell
      heading={`Planning`}
      subheading={site.name}
      userEmail={user.email}
      primarySite={{ id: site.id, name: site.name }}
      actions={<GeneratePlanningButton siteId={site.id} />}
    >
      {planningTasks.length > 0 ? (
        <div className="-mx-4 -mt-8 lg:-mx-10 lg:-mt-8" style={{ height: 'calc(100vh - 120px)' }}>
          <PlanningViewEditorial
            siteId={site.id}
            siteName={site.name}
            siteAddress={site.address ?? null}
            siteDeadline={site.deadline ?? null}
            initialTasks={planningTasks}
            workers={planningWorkers}
          />
        </div>
      ) : (
        <div className="border border-dashed border-rule bg-paper p-12 text-center">
          <p className="font-mono text-[11px] uppercase tracking-widest text-ink-3 mb-6">
            Aucun planning généré
          </p>
          <p className="text-[15px] text-ink-2 mb-8 max-w-sm mx-auto">
            Générez un planning IA à partir des tâches de ce chantier.
          </p>
          <GeneratePlanningButton siteId={site.id} />
        </div>
      )}
    </AppShell>
  );
}
