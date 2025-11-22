import { notFound, redirect } from 'next/navigation';
import Link from 'next/link';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { AddTaskForm } from '@/app/site/[id]/add-task-form';
import { CompleteTaskButton } from '@/app/site/[id]/complete-task-button';

type Params = {
  params: Promise<{
    id: string;
  }>;
};

function isValidUuid(value: string | undefined) {
  return Boolean(value && /^[0-9a-fA-F-]{36}$/.test(value));
}

export default async function SiteTasksPage({ params }: Params) {
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

  // Récupérer les tâches du chantier
  const { data: tasks } = await supabase
    .from('tasks')
    .select('id, title, status, required_role, duration_hours, created_at')
    .eq('site_id', site.id)
    .order('created_at', { ascending: true });

  const pendingTasks = tasks?.filter((task) => task.status !== 'done') ?? [];
  const doneTasks = tasks?.filter((task) => task.status === 'done') ?? [];

  return (
    <AppShell
      heading={`Tâches - ${site.name}`}
      subheading={`Toutes les tâches du chantier ${site.name}`}
      userEmail={user.email}
      primarySite={{ id: site.id, name: site.name }}
    >
      <div className="space-y-6">
        {/* Stats */}
        <section className="grid gap-4 md:grid-cols-3">
          <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
            <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Total</p>
            <p className="mt-2 text-3xl font-semibold">{tasks?.length ?? 0}</p>
            <p className="text-sm text-zinc-500">tâches</p>
          </div>
          <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5 dark:border-amber-500/30 dark:bg-amber-900/20">
            <p className="text-xs uppercase tracking-[0.3em] text-amber-800 dark:text-amber-200">
              À traiter
            </p>
            <p className="mt-2 text-3xl font-semibold text-amber-900 dark:text-amber-100">
              {pendingTasks.length}
            </p>
            <p className="text-sm text-amber-800/80 dark:text-amber-200">en attente</p>
          </div>
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-5 dark:border-emerald-500/30 dark:bg-emerald-900/20">
            <p className="text-xs uppercase tracking-[0.3em] text-emerald-800 dark:text-emerald-100">
              Terminées
            </p>
            <p className="mt-2 text-3xl font-semibold text-emerald-900 dark:text-emerald-50">
              {doneTasks.length}
            </p>
            <p className="text-sm text-emerald-800/80 dark:text-emerald-100">clôturées</p>
          </div>
        </section>

        {/* Ajouter une tâche */}
        <section className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div className="mb-4">
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Ajouter une tâche
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Créez une nouvelle tâche pour ce chantier.
            </p>
          </div>
          <AddTaskForm siteId={site.id} />
        </section>

        {/* Tâches en cours */}
        <section className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div className="mb-4">
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Tâches en cours
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Tâches en attente de réalisation.
            </p>
          </div>
          {pendingTasks.length ? (
            <div className="space-y-3">
              {pendingTasks.map((task) => (
                <div
                  key={task.id}
                  className="rounded-2xl border border-zinc-200 p-4 dark:border-zinc-700"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                        {task.title}
                      </p>
                      <p className="text-xs text-zinc-500 dark:text-zinc-400">
                        {task.required_role || 'Rôle libre'}
                        {task.duration_hours
                          ? ` • Durée estimée ${task.duration_hours}h`
                          : ''}
                      </p>
                    </div>
                    <CompleteTaskButton siteId={site.id} taskId={task.id} />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="rounded-2xl border border-dashed border-zinc-200 p-6 text-center text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
              Aucune tâche en attente pour le moment.
            </p>
          )}
        </section>

        {/* Historique */}
        <section className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
            Historique récent
          </h2>
          {doneTasks.length ? (
            <ul className="mt-4 grid gap-3 md:grid-cols-2">
              {doneTasks.slice(0, 6).map((task) => (
                <li
                  key={task.id}
                  className="rounded-2xl border border-zinc-200 p-4 text-sm text-zinc-600 dark:border-zinc-700 dark:text-zinc-300"
                >
                  ✅ <span className="font-semibold text-zinc-900 dark:text-white">{task.title}</span>{' '}
                  terminé
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-sm text-zinc-500 dark:text-zinc-400">
              Aucune tâche clôturée récemment.
            </p>
          )}
        </section>
      </div>
    </AppShell>
  );
}

