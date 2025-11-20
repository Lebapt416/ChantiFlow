import Image from 'next/image';
import Link from 'next/link';
import { redirect, notFound } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';

type Params = {
  params: Promise<{
    siteId: string;
  }>;
};

export default async function ReportPage({ params }: Params) {
  const { siteId } = await params;

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
    .eq('id', siteId)
    .single();

  if (siteError || !site || site.created_by !== user.id) {
    notFound();
  }

  const [{ data: tasks }, { data: reportsData }] = await Promise.all([
    supabase
      .from('tasks')
      .select('id, title, status, required_role, duration_hours')
      .eq('site_id', site.id)
      .order('created_at', { ascending: true }),
    supabase
      .from('reports')
      .select('id, task_id, worker_id, description, photo_url, created_at')
      .order('created_at', { ascending: false }),
  ]);

  const taskById =
    tasks?.reduce<Record<string, (typeof tasks)[number]>>((acc, task) => {
      acc[task.id] = task;
      return acc;
    }, {}) ?? {};

  const taskIds = tasks?.map((task) => task.id) ?? [];
  const filteredReports =
    reportsData?.filter((report) => taskIds.includes(report.task_id ?? '')) ??
    [];

  const { data: workers } = await supabase
    .from('workers')
    .select('id, name, role, email')
    .eq('site_id', site.id);

  const workerById =
    workers?.reduce<Record<string, (typeof workers)[number]>>(
      (acc, worker) => {
        acc[worker.id] = worker;
        return acc;
      },
      {},
    ) ?? {};

  const totalTasks = tasks?.length ?? 0;
  const doneTasks = tasks?.filter((task) => task.status === 'done').length ?? 0;
  const progress = totalTasks > 0 ? Math.round((doneTasks / totalTasks) * 100) : 0;

  return (
    <AppShell
      heading={`Rapports – ${site.name}`}
      subheading={`Progression globale : ${progress}% (${doneTasks}/${totalTasks} tâches)`} 
      userEmail={user.email}
      primarySite={{ id: site.id, name: site.name }}
      actions={
        <Link
          href={`/site/${site.id}`}
          className="rounded-full border border-zinc-200 px-4 py-2 text-sm font-medium text-zinc-700 transition hover:border-zinc-900 hover:text-zinc-900 dark:border-zinc-700 dark:text-zinc-200 dark:hover:border-white dark:hover:text-white"
        >
          ← Retour fiche
        </Link>
      }
    >
      <section className="mb-6 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Dernier rapport reçu :{' '}
          {filteredReports[0]?.created_at
            ? new Date(filteredReports[0].created_at ?? '').toLocaleString('fr-FR')
            : '—'}
        </p>
        <div className="mt-4 h-2 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
          <div
            className="h-full rounded-full bg-emerald-500 transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[1fr_0.7fr]">
        <section className="space-y-4 rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">
            Rapports reçus
          </h2>
          {filteredReports.length ? (
            <ul className="space-y-4">
              {filteredReports.map((report) => {
                const task = taskById[report.task_id ?? ''];
                const worker = workerById[report.worker_id ?? ''];
                return (
                  <li
                    key={report.id}
                    className="rounded-xl border border-zinc-100 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                        {task?.title ?? 'Tâche inconnue'}
                      </p>
                      <span className="text-xs text-zinc-500 dark:text-zinc-400">
                        {new Date(report.created_at ?? '').toLocaleString(
                          'fr-FR',
                          {
                            day: '2-digit',
                            month: 'short',
                            hour: '2-digit',
                            minute: '2-digit',
                          },
                        )}
                      </span>
                    </div>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      {worker?.name ?? 'Employé inconnu'} •{' '}
                      {worker?.role ?? 'Rôle non renseigné'} •{' '}
                      {worker?.email ?? 'Email inconnu'}
                    </p>
                    {report.description ? (
                      <p className="mt-3 text-sm text-zinc-700 dark:text-zinc-200">
                        {report.description}
                      </p>
                    ) : null}
                    {report.photo_url ? (
                      <div className="mt-3 overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-700">
                        <Image
                          src={report.photo_url}
                          alt="Photo de progression"
                          width={800}
                          height={450}
                          className="h-64 w-full object-cover"
                        />
                      </div>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="rounded-xl border border-dashed border-zinc-200 bg-zinc-50 p-6 text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
              Aucun rapport pour le moment.
            </p>
          )}
        </section>

        <section className="space-y-4 rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">
            Détails des tâches
          </h2>
          <ul className="space-y-3">
            {tasks?.length ? (
              tasks.map((task) => (
                <li
                  key={task.id}
                  className="rounded-xl border border-zinc-100 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900"
                >
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                      {task.title}
                    </p>
                    <span
                      className={`rounded-full px-3 py-1 text-xs font-semibold ${
                        task.status === 'done'
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-amber-100 text-amber-800'
                      }`}
                    >
                      {task.status === 'done' ? 'Terminé' : 'En cours'}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    {task.required_role || 'Rôle libre'} •{' '}
                    {task.duration_hours
                      ? `${task.duration_hours}h`
                      : 'Durée non renseignée'}
                  </p>
                </li>
              ))
            ) : (
              <p className="rounded-xl border border-dashed border-zinc-200 bg-zinc-50 p-6 text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
                Ajoute des tâches dans l’onglet chantier.
              </p>
            )}
          </ul>
        </section>
      </div>
    </AppShell>
  );
}

