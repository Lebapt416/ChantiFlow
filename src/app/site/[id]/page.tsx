import Link from 'next/link';
import { notFound, redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AddTaskForm } from './add-task-form';
import { AddWorkerForm } from './add-worker-form';
import { CompleteTaskButton } from './complete-task-button';
import { SiteQrCard } from './qr-card';
import { AppShell } from '@/components/app-shell';

 type Params = {
  params: Promise<{
    id: string;
  }>;
};

function isValidUuid(value: string | undefined) {
  return Boolean(value && /^[0-9a-fA-F-]{36}$/.test(value));
}

export default async function SitePage({ params }: Params) {
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

  if (siteError) {
    return (
      <div className="min-h-screen bg-zinc-50 px-6 py-10 dark:bg-zinc-950">
        <div className="mx-auto max-w-2xl rounded-2xl border border-rose-200 bg-white p-8 text-center shadow-lg dark:border-rose-900/60 dark:bg-zinc-900">
          <p className="text-sm uppercase tracking-[0.3em] text-rose-500">Erreur</p>
          <h1 className="mt-2 text-2xl font-semibold text-zinc-900 dark:text-white">
            Impossible de charger ce chantier
          </h1>
          <p className="mt-4 text-sm text-zinc-500 dark:text-zinc-400">{siteError.message}</p>
          <Link
            href="/dashboard"
            className="mt-6 inline-flex rounded-full bg-black px-5 py-2 text-sm font-medium text-white dark:bg-white dark:text-black"
          >
            Retour au dashboard
          </Link>
        </div>
      </div>
    );
  }

  if (!site) {
    notFound();
  }

  if (site.created_by && site.created_by !== user.id) {
    notFound();
  }

  const [{ data: tasks }, { data: workers }] = await Promise.all([
    supabase
      .from('tasks')
      .select('id, title, status, required_role, duration_hours, created_at')
      .eq('site_id', site.id)
      .order('created_at', { ascending: true }),
    supabase
      .from('workers')
      .select('id, name, email, role, created_at')
      .eq('site_id', site.id)
      .order('created_at', { ascending: true }),
  ]);

  const appUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? '';
  const qrUrl = `${appUrl.replace(/\/$/, '')}/qr/${site.id}`;
  const totalTasks = tasks?.length ?? 0;
  const doneTasks = tasks?.filter((task) => task.status === 'done').length ?? 0;
  const workerCount = workers?.length ?? 0;

  return (
    <AppShell
      heading={site.name}
      subheading={
        site.deadline
          ? `Deadline ${new Date(site.deadline).toLocaleDateString('fr-FR')}`
          : 'Deadline à définir'
      }
      userEmail={user.email}
      primarySite={{ id: site.id, name: site.name }}
      actions={
        <div className="flex gap-2">
          <Link
            href={`/qr/${site.id}`}
            className="rounded-full border border-zinc-200 px-4 py-2 text-sm font-medium text-zinc-700 transition hover:border-zinc-900 hover:text-zinc-900 dark:border-zinc-700 dark:text-zinc-200 dark:hover:border-white dark:hover:text-white"
          >
            QR employé
          </Link>
          <Link
            href={`/report/${site.id}`}
            className="rounded-full border border-zinc-200 px-4 py-2 text-sm font-medium text-zinc-700 transition hover:border-zinc-900 hover:text-zinc-900 dark:border-zinc-700 dark:text-zinc-200 dark:hover:border-white dark:hover:text-white"
          >
            Rapports
          </Link>
        </div>
      }
    >
      <nav className="mb-6 flex gap-3 overflow-x-auto rounded-2xl border border-zinc-100 bg-white p-3 text-xs font-semibold uppercase tracking-[0.3em] text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
        <a href="#tasks" className="rounded-full border border-transparent px-3 py-1 hover:border-zinc-400">
          Tâches
        </a>
        <a href="#team" className="rounded-full border border-transparent px-3 py-1 hover:border-zinc-400">
          Équipe
        </a>
        <a href="#timeline" className="rounded-full border border-transparent px-3 py-1 hover:border-zinc-400">
          Timeline
        </a>
        <a href="#qr" className="rounded-full border border-transparent px-3 py-1 hover:border-zinc-400">
          QR & Accès
        </a>
      </nav>

      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Tâches</p>
          <p className="mt-2 text-3xl font-semibold">
            {doneTasks}/{totalTasks}
          </p>
          <p className="text-sm text-zinc-500">terminées</p>
        </div>
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Équipe</p>
          <p className="mt-2 text-3xl font-semibold">{workerCount}</p>
          <p className="text-sm text-zinc-500">collaborateurs inscrits</p>
        </div>
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Planning IA</p>
          <p className="mt-2 text-base text-zinc-500">
            Disponible bientôt — prépare tes tâches pour l’automatisation.
          </p>
        </div>
      </section>

      <div className="mt-8 grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          <section
            id="tasks"
            className="space-y-4 rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
                  Tâches
                </p>
                <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">
                  Plan de travail
                </h2>
              </div>
            </div>

            <AddTaskForm siteId={site.id} />

            <div className="space-y-3">
              {tasks?.length ? (
                tasks.map((task) => (
                  <div
                    key={task.id}
                    className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-zinc-100 bg-zinc-50 px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900"
                  >
                    <div>
                      <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                        {task.title}
                      </p>
                      <p className="text-xs text-zinc-500 dark:text-zinc-400">
                        {task.required_role || 'Rôle libre'} •{' '}
                        {task.duration_hours ? `${task.duration_hours}h` : 'Durée estimée ?'}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className={`rounded-full px-3 py-1 text-xs font-semibold ${
                          task.status === 'done'
                            ? 'bg-emerald-100 text-emerald-800'
                            : 'bg-amber-100 text-amber-800'
                        }`}
                      >
                        {task.status === 'done' ? 'Terminé' : 'En attente'}
                      </span>
                      <CompleteTaskButton
                        siteId={site.id}
                        taskId={task.id}
                        disabled={task.status === 'done'}
                      />
                    </div>
                  </div>
                ))
              ) : (
                <p className="rounded-xl border border-dashed border-zinc-200 bg-white p-6 text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
                  Ajoute tes premières tâches pour générer un planning.
                </p>
              )}
            </div>
          </section>

          <section
            id="team"
            className="space-y-4 rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
                  Équipe
                </p>
                <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">
                  Employés sur site
                </h2>
              </div>
            </div>

            <AddWorkerForm siteId={site.id} />

            <div className="space-y-3">
              {workers?.length ? (
                workers.map((worker) => (
                  <div
                    key={worker.id}
                    className="rounded-xl border border-zinc-100 bg-zinc-50 px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900"
                  >
                    <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                      {worker.name}
                    </p>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      {worker.role ?? 'Rôle non défini'}
                    </p>
                    {worker.email ? (
                      <p className="text-xs text-zinc-400 dark:text-zinc-500">{worker.email}</p>
                    ) : null}
                  </div>
                ))
              ) : (
                <p className="rounded-xl border border-dashed border-zinc-200 bg-white p-6 text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
                  Aucun employé enregistré pour ce chantier.
                </p>
              )}
            </div>
          </section>

          <section
            id="timeline"
            className="space-y-4 rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
                  Timeline
                </p>
                <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">
                  Chronologie du chantier
                </h2>
              </div>
            </div>
            {tasks?.length ? (
              <ol className="space-y-4 border-l border-zinc-200 pl-4 dark:border-zinc-700">
                {tasks.map((task) => (
                  <li key={task.id} className="relative">
                    <span className="absolute -left-[9px] top-1 h-3 w-3 rounded-full bg-zinc-900 dark:bg-white" />
                    <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                      {task.title}
                    </p>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      {task.status === 'done' ? 'Terminé' : 'En cours'} •{' '}
                      {task.created_at
                        ? new Date(task.created_at).toLocaleDateString('fr-FR', {
                            day: '2-digit',
                            month: 'short',
                          })
                        : 'date inconnue'}
                    </p>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Ajoute des tâches pour alimenter la chronologie.
              </p>
            )}
          </section>
        </div>

        <div className="space-y-6">
          <SiteQrCard siteName={site.name} targetUrl={qrUrl} />
          <div
            id="qr"
            className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900"
          >
            <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
              Accès employé
            </p>
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">
              QR code à afficher
            </h2>
            <p className="mt-3 text-sm text-zinc-500 dark:text-zinc-400">
              Affiche ce QR sur le chantier pour que les équipes accèdent à l’interface `/qr/{'{siteId}'}` sans authentification.
            </p>
            <Link href={`/qr/${site.id}`} className="mt-4 inline-flex text-sm font-medium text-black dark:text-white">
              Ouvrir l’interface employé →
            </Link>
            <Link
              href={`/report/${site.id}`}
              className="mt-2 inline-flex text-sm font-medium text-zinc-700 dark:text-zinc-300"
            >
              Voir les rapports →
            </Link>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
