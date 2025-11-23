import { redirect } from 'next/navigation';
import Link from 'next/link';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AddTaskForm } from './add-task-form';
import { AssignTaskButton } from './assign-task-button';

export const metadata = {
  title: 'Tâches | ChantiFlow',
};

export default async function TasksPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: sites } = await supabase
    .from('sites')
    .select('id, name, deadline')
    .eq('created_by', user.id);

  const siteMap =
    sites?.reduce<Record<string, { name: string; deadline: string | null }>>(
      (acc, site) => {
        acc[site.id] = { name: site.name, deadline: site.deadline };
        return acc;
      },
      {},
    ) ?? {};

  const siteIds = Object.keys(siteMap);

  // Récupérer les tâches avec les workers assignés
  const { data: tasks } = siteIds.length
    ? await supabase
        .from('tasks')
        .select(
          'id, site_id, title, status, required_role, duration_hours, created_at, assigned_worker_id',
        )
        .in('site_id', siteIds)
        .order('created_at', { ascending: true })
    : { data: [] };

  // Récupérer tous les workers de tous les chantiers de l'utilisateur
  let availableWorkers: Array<{
    id: string;
    name: string;
    email: string | null;
    role: string | null;
  }> = [];

  if (siteIds.length > 0) {
    // Récupérer les workers assignés aux chantiers
    const { data: siteWorkers } = await supabase
      .from('workers')
      .select('id, name, email, role, site_id, created_by')
      .in('site_id', siteIds);

    // Récupérer les workers au niveau du compte (sans site_id)
    const { data: accountWorkers } = await supabase
      .from('workers')
      .select('id, name, email, role, created_by')
      .eq('created_by', user.id)
      .is('site_id', null);

    // Combiner et dédupliquer
    const allWorkers = [
      ...(siteWorkers ?? []),
      ...(accountWorkers ?? []),
    ];

    // Filtrer pour ne garder que ceux de l'utilisateur et dédupliquer par ID
    const workerMap = new Map<string, typeof availableWorkers[0]>();
    allWorkers.forEach((worker) => {
      if (worker.created_by === user.id && !workerMap.has(worker.id)) {
        workerMap.set(worker.id, {
          id: worker.id,
          name: worker.name,
          email: worker.email,
          role: worker.role,
        });
      }
    });
    availableWorkers = Array.from(workerMap.values());
  }

  const pendingTasks = tasks?.filter((task) => task.status !== 'done') ?? [];
  const doneTasks = tasks?.filter((task) => task.status === 'done') ?? [];

  return (
    <AppShell
      heading="Tâches"
      subheading="Vue consolidée de toutes les tâches à travers tes chantiers."
      userEmail={user.email}
      primarySite={sites?.[0] ?? null}
    >
      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Total</p>
          <p className="mt-2 text-3xl font-semibold">{tasks?.length ?? 0}</p>
          <p className="text-sm text-zinc-500">tâches recensées</p>
        </div>
        <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5 dark:border-amber-500/30 dark:bg-amber-900/20">
          <p className="text-xs uppercase tracking-[0.3em] text-amber-800 dark:text-amber-200">
            À traiter
          </p>
          <p className="mt-2 text-3xl font-semibold text-amber-900 dark:text-amber-100">
            {pendingTasks.length}
          </p>
          <p className="text-sm text-amber-800/80 dark:text-amber-200">
            tâches en attente
          </p>
        </div>
        <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-5 dark:border-emerald-500/30 dark:bg-emerald-900/20">
          <p className="text-xs uppercase tracking-[0.3em] text-emerald-800 dark:text-emerald-100">
            Clôturées
          </p>
          <p className="mt-2 text-3xl font-semibold text-emerald-900 dark:text-emerald-50">
            {doneTasks.length}
          </p>
          <p className="text-sm text-emerald-800/80 dark:text-emerald-100">
            tâches terminées
          </p>
        </div>
      </section>

      <section className="mt-8 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Ajouter une tâche
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Créez une nouvelle tâche pour un chantier.
            </p>
          </div>
        </div>
        <div className="mb-8">
          <AddTaskForm sites={sites ?? []} />
        </div>
      </section>

      <section className="mt-8 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Tâches en cours
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Filtre automatiquement par statut & site.
            </p>
          </div>
        </div>
        {pendingTasks.length ? (
          <div className="space-y-3">
            {pendingTasks.map((task) => (
              <div
                key={task.id}
                className="rounded-2xl border border-zinc-200 p-4 dark:border-zinc-700"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                      {task.title}
                    </p>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      {siteMap[task.site_id]?.name ?? 'Site inconnu'} •{' '}
                      {task.required_role || 'Rôle libre'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <AssignTaskButton
                      taskId={task.id}
                      siteId={task.site_id}
                      currentWorkerId={(task as any).assigned_worker_id || null}
                      availableWorkers={availableWorkers}
                    />
                    <Link
                      href={`/site/${task.site_id}`}
                      className="text-xs font-semibold text-black hover:underline dark:text-white"
                    >
                      Voir le chantier →
                    </Link>
                  </div>
                </div>
                <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
                  Créée le{' '}
                  {task.created_at
                    ? new Date(task.created_at).toLocaleDateString('fr-FR', {
                        day: '2-digit',
                        month: 'short',
                      })
                    : '---'}
                  {task.duration_hours
                    ? ` • Durée estimée ${task.duration_hours}h`
                    : ''}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="rounded-2xl border border-dashed border-zinc-200 p-6 text-center text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
            Aucune tâche en attente pour le moment.
          </p>
        )}
      </section>

      <section className="mt-8 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
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
                terminé sur {siteMap[task.site_id]?.name ?? 'site inconnu'}
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-sm text-zinc-500 dark:text-zinc-400">
            Aucune tâche clôturée récemment.
          </p>
        )}
      </section>
    </AppShell>
  );
}

