import Link from 'next/link';
import Image from 'next/image';
import { redirect } from 'next/navigation';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export const metadata = {
  title: 'Rapports | ChantiFlow',
};

export default async function ReportsHubPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: sites } = await supabase
    .from('sites')
    .select('id, name')
    .eq('created_by', user.id);

  const siteMap =
    sites?.reduce<Record<string, string>>((acc, site) => {
      acc[site.id] = site.name;
      return acc;
    }, {}) ?? {};

  const siteIds = Object.keys(siteMap);

  const { data: reports } = siteIds.length
    ? await supabase
        .from('reports')
        .select('id, task_id, worker_id, photo_url, description, created_at')
        .order('created_at', { ascending: false })
        .limit(20)
    : { data: [] };

  const { data: tasks } = siteIds.length
    ? await supabase
        .from('tasks')
        .select('id, title, site_id')
        .in('site_id', siteIds)
    : { data: [] };

  const { data: workers } = siteIds.length
    ? await supabase
        .from('workers')
        .select('id, name, role')
        .in('site_id', siteIds)
    : { data: [] };

  const taskMap =
    tasks?.reduce<Record<string, { title: string; site_id: string }>>(
      (acc, task) => {
        acc[task.id] = { title: task.title, site_id: task.site_id };
        return acc;
      },
      {},
    ) ?? {};

  const workerMap =
    workers?.reduce<Record<string, { name: string; role: string | null }>>(
      (acc, worker) => {
        acc[worker.id] = { name: worker.name, role: worker.role };
        return acc;
      },
      {},
    ) ?? {};

  return (
    <AppShell
      heading="Rapports terrain"
      subheading="Toutes les remontées photo/texte envoyées via les QR codes."
      userEmail={user.email}
      primarySite={sites?.[0] ?? null}
    >
      <section className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Derniers rapports
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Derniers 20 envois depuis les chantiers.
            </p>
          </div>
          {sites?.length ? (
            <Link
              href={`/report/${sites[0].id}`}
              className="rounded-full border border-zinc-200 px-4 py-2 text-sm font-medium text-zinc-700 transition hover:border-zinc-900 hover:text-zinc-900 dark:border-zinc-700 dark:text-zinc-200 dark:hover:border-white dark:hover:text-white"
            >
              Ouvrir un chantier →
            </Link>
          ) : null}
        </div>
        {reports?.length ? (
          <div className="space-y-4">
            {reports.map((report) => {
              const task = taskMap[report.task_id ?? ''];
              const worker = workerMap[report.worker_id ?? ''];
              return (
                <div
                  key={report.id}
                  className="rounded-2xl border border-zinc-200 p-4 dark:border-zinc-700"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                        {task?.title ?? 'Tâche inconnue'}
                      </p>
                      <p className="text-xs text-zinc-500 dark:text-zinc-400">
                        {siteMap[task?.site_id ?? ''] ?? 'Site inconnu'}
                      </p>
                    </div>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      {new Date(report.created_at ?? '').toLocaleString('fr-FR')}
                    </p>
                  </div>
                  <p className="mt-2 text-xs text-zinc-400 dark:text-zinc-500">
                    {worker?.name ?? 'Employé inconnu'} •{' '}
                    {worker?.role ?? 'Rôle non renseigné'}
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
                        alt="Photo rapport"
                        width={800}
                        height={320}
                        className="h-48 w-full object-cover"
                      />
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Aucun rapport pour le moment.
          </p>
        )}
      </section>

      <section className="mt-8 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
          Accès QR disponibles
        </h2>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {sites?.length ? (
            sites.map((site) => (
              <Link
                key={site.id}
                href={`/qr/${site.id}`}
                className="rounded-2xl border border-zinc-200 p-4 text-sm font-semibold transition hover:border-zinc-900 dark:border-zinc-700 dark:hover:border-white"
              >
                QR – {site.name}
              </Link>
            ))
          ) : (
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Crée un chantier pour générer des QR codes.
            </p>
          )}
        </div>
      </section>
    </AppShell>
  );
}

