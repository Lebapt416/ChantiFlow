import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { WeeklyCalendar } from '@/components/weekly-calendar';
import { GeneratePlanningButton } from '@/app/site/[id]/generate-planning-button';
import Link from 'next/link';
import { SiteSelector } from '@/components/site-selector';

export const metadata = {
  title: 'Planning | ChantiFlow',
};

type SearchParams = {
  searchParams: Promise<{
    site?: string;
  }>;
};

export default async function PlanningPage({ searchParams }: SearchParams) {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  // Récupérer tous les chantiers
  const { data: sites } = await supabase
    .from('sites')
    .select('id, name, deadline')
    .eq('created_by', user.id)
    .order('created_at', { ascending: false });

  if (!sites || sites.length === 0) {
    return (
      <AppShell
        heading="Planning"
        subheading="Visualisez le planning de vos chantiers"
        userEmail={user.email}
        primarySite={null}
      >
        <div className="rounded-2xl border border-zinc-200 bg-white p-8 text-center dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Aucun chantier disponible. Créez un chantier pour voir son planning.
          </p>
          <Link
            href="/dashboard"
            className="mt-4 inline-block rounded-lg bg-black px-4 py-2 text-sm font-semibold text-white transition hover:bg-zinc-800 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
          >
            Créer un chantier
          </Link>
        </div>
      </AppShell>
    );
  }

  // Récupérer le site sélectionné depuis les query params ou utiliser le premier
  const params = await searchParams;
  const selectedSiteId = params?.site || sites[0].id;
  const currentSite = sites.find((s) => s.id === selectedSiteId) || sites[0];

  // Récupérer les tâches et workers du chantier
  const [{ data: tasks }, { data: workers }] = await Promise.all([
    supabase
      .from('tasks')
      .select(
        'id, title, required_role, duration_hours, status, planned_start, planned_end, planned_worker_id, planned_order',
      )
      .eq('site_id', currentSite.id),
    supabase
      .from('workers')
      .select('id, name, email, role')
      .eq('site_id', currentSite.id),
  ]);

  const planning =
    tasks
      ?.filter((task) => task.planned_start && task.planned_end)
      .map((task) => ({
        taskId: task.id,
        taskTitle: task.title,
        order: task.planned_order ?? 0,
        startDate: task.planned_start as string,
        endDate: task.planned_end as string,
        assignedWorkerId: task.planned_worker_id,
      }))
      .sort((a, b) => a.order - b.order)
      .map((task, index, arr) => ({
        ...task,
        priority:
          index === 0 ? ('high' as const) : index >= arr.length - 2 ? ('low' as const) : ('medium' as const),
      })) ?? null;

  return (
    <AppShell
      heading="Planning IA"
      subheading={`Planning optimisé par IA pour le chantier ${currentSite.name}`}
      userEmail={user.email}
      primarySite={{ id: currentSite.id, name: currentSite.name }}
      actions={
        sites.length > 1 ? (
          <SiteSelector sites={sites} currentSiteId={currentSite.id} />
        ) : null
      }
    >
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">
              {currentSite.name}
            </h2>
            {currentSite.deadline && (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Deadline: {new Date(currentSite.deadline).toLocaleDateString('fr-FR')}
              </p>
            )}
          </div>
          <GeneratePlanningButton siteId={currentSite.id} />
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
            <GeneratePlanningButton siteId={currentSite.id} />
          </div>
        )}
      </div>
    </AppShell>
  );
}

