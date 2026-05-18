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
    try {
      const { data: siteWorkers, error: siteWorkersError } = await supabase
        .from('workers')
        .select('id, name, email, role, site_id, created_by')
        .in('site_id', siteIds);

      if (siteWorkersError) {
        console.error('Erreur récupération workers des chantiers:', siteWorkersError);
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let accountWorkers: any[] = [];
      try {
        const { data: accountWorkersData, error: accountWorkersError } = await supabase
          .from('workers')
          .select('id, name, email, role, created_by')
          .eq('created_by', user.id)
          .is('site_id', null);

        if (accountWorkersError) {
          if (accountWorkersError.message.includes('created_by') || accountWorkersError.code === '42703') {
            console.warn('Colonne created_by non trouvée, récupération sans filtre');
            const { data: allWorkersData } = await supabase
              .from('workers')
              .select('id, name, email, role')
              .is('site_id', null);
            accountWorkers = allWorkersData ?? [];
          }
        } else {
          accountWorkers = accountWorkersData ?? [];
        }
      } catch (error) {
        console.error('Erreur récupération workers du compte:', error);
      }

      const allWorkers = [
        ...(siteWorkers ?? []),
        ...accountWorkers,
      ];

      const workerMap = new Map<string, typeof availableWorkers[0]>();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      allWorkers.forEach((worker: any) => {
        if (!worker.created_by || worker.created_by === user.id) {
          if (!workerMap.has(worker.id)) {
            workerMap.set(worker.id, {
              id: worker.id,
              name: worker.name,
              email: worker.email,
              role: worker.role,
            });
          }
        }
      });
      availableWorkers = Array.from(workerMap.values());
    } catch (error) {
      console.error('Erreur lors de la récupération des workers:', error);
    }
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
        <div className="rounded border border-rule-soft bg-paper p-5 dark:border-rule dark:bg-ink">
          <p className="text-xs uppercase tracking-[0.3em] text-ink-3">Total</p>
          <p className="mt-2 text-3xl font-semibold text-ink dark:text-paper">{tasks?.length ?? 0}</p>
          <p className="text-sm text-ink-3">tâches recensées</p>
        </div>
        <div className="rounded border border-amber-200 bg-amber-50 p-5 dark:border-amber-500/30 dark:bg-amber-900/20">
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
        <div className="rounded border border-rule-soft bg-paper-2 p-5 dark:border-orange/30 dark:bg-paper-2">
          <p className="text-xs uppercase tracking-[0.3em] text-ink dark:text-paper">
            Clôturées
          </p>
          <p className="mt-2 text-3xl font-semibold text-ink dark:text-orange">
            {doneTasks.length}
          </p>
          <p className="text-sm text-ink/80 dark:text-paper">
            tâches terminées
          </p>
        </div>
      </section>

      <section className="mt-8 rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="font-serif text-[22px] text-ink dark:text-paper">
              Ajouter une tâche
            </h2>
            <p className="text-sm text-ink-3">
              Créez une nouvelle tâche pour un chantier.
            </p>
          </div>
        </div>
        <div className="mb-8">
          <AddTaskForm sites={sites ?? []} />
        </div>
      </section>

      <section className="mt-8 rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="font-serif text-[22px] text-ink dark:text-paper">
              Tâches en cours
            </h2>
            <p className="text-sm text-ink-3">
              Filtre automatiquement par statut & site.
            </p>
          </div>
        </div>
        {pendingTasks.length ? (
          <div className="space-y-3">
            {pendingTasks.map((task) => {
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              const assignedWorkerId = (task as any).assigned_worker_id || null;
              return (
                <div
                  key={task.id}
                  className="rounded border border-rule-soft p-4 dark:border-rule"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold text-ink dark:text-paper">
                        {task.title}
                      </p>
                      <p className="text-xs text-ink-3">
                        {siteMap[task.site_id]?.name ?? 'Site inconnu'} •{' '}
                        {task.required_role || 'Rôle libre'}
                      </p>
                    </div>
                    <div className="flex items-center gap-3 flex-shrink-0">
                      <AssignTaskButton
                        taskId={task.id}
                        siteId={task.site_id}
                        currentWorkerId={assignedWorkerId}
                        availableWorkers={availableWorkers}
                      />
                      <Link
                        href={`/site/${task.site_id}`}
                        className="font-mono text-[10px] uppercase tracking-widest text-ink-2 hover:text-ink whitespace-nowrap"
                      >
                        Voir le chantier →
                      </Link>
                    </div>
                  </div>
                  {availableWorkers.length === 0 && (
                    <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
                      ⚠️ Aucun membre d&apos;équipe disponible. Ajoutez des membres dans la page &quot;Équipe&quot;.
                    </p>
                  )}
                  <p className="mt-2 text-xs text-ink-3">
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
              );
            })}
          </div>
        ) : (
          <p className="rounded border border-dashed border-rule-soft p-6 text-center text-sm text-ink-3 dark:border-rule">
            Aucune tâche en attente pour le moment.
          </p>
        )}
      </section>

      <section className="mt-8 rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
        <h2 className="font-serif text-[22px] text-ink dark:text-paper">
          Historique récent
        </h2>
        {doneTasks.length ? (
          <ul className="mt-4 grid gap-3 md:grid-cols-2">
            {doneTasks.slice(0, 6).map((task) => (
              <li
                key={task.id}
                className="rounded border border-rule-soft p-4 text-sm text-ink-2 dark:border-rule"
              >
                ✅ <span className="font-semibold text-ink dark:text-paper">{task.title}</span>{' '}
                terminé sur {siteMap[task.site_id]?.name ?? 'site inconnu'}
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-sm text-ink-3">
            Aucune tâche clôturée récemment.
          </p>
        )}
      </section>
    </AppShell>
  );
}
