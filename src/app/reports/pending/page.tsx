import { redirect } from 'next/navigation';
import Link from 'next/link';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { PendingReportsList } from '../pending-reports-list';
import { ArrowLeft } from 'lucide-react';

export const metadata = {
  title: 'Rapports en attente | ChantiFlow',
};

export default async function PendingReportsPage() {
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

  // Récupérer tous les rapports
  const { data: allReports } = siteIds.length
    ? await supabase
        .from('reports')
        .select('id, task_id, worker_id, photo_url, description, created_at')
        .order('created_at', { ascending: false })
    : { data: [] };

  // Récupérer le statut des tâches pour distinguer les rapports validés
  const { data: tasksWithStatus } = siteIds.length
    ? await supabase
        .from('tasks')
        .select('id, title, site_id, status')
        .in('site_id', siteIds)
    : { data: [] };

  const { data: tasks } = siteIds.length
    ? await supabase
        .from('tasks')
        .select('id, title, site_id')
        .in('site_id', siteIds)
    : { data: [] };

  // Récupérer les workers de deux façons :
  // 1. Workers actuellement sur les chantiers
  // 2. Workers historiques via les rapports (pour les chantiers terminés)
  const { data: workersFromSites } = siteIds.length
    ? await supabase
        .from('workers')
        .select('id, name, role')
        .in('site_id', siteIds)
    : { data: [] };

  // Récupérer les worker_ids depuis les rapports pour les workers historiques
  const workerIdsFromReports = [
    ...new Set(
      allReports
        ?.map((r) => r.worker_id)
        .filter((id): id is string => id !== null) ?? [],
    ),
  ];

  const { data: workersFromReports } =
    workerIdsFromReports.length > 0
      ? await supabase
          .from('workers')
          .select('id, name, role')
          .in('id', workerIdsFromReports)
      : { data: [] };

  // Fusionner les deux listes de workers (en évitant les doublons)
  const allWorkers = [
    ...(workersFromSites ?? []),
    ...(workersFromReports ?? []),
  ];
  const uniqueWorkers = Array.from(
    new Map(allWorkers.map((w) => [w.id, w])).values(),
  );

  const taskMap =
    tasks?.reduce<Record<string, { title: string; site_id: string }>>(
      (acc, task) => {
        acc[task.id] = { title: task.title, site_id: task.site_id };
        return acc;
      },
      {},
    ) ?? {};

  const taskStatusMap =
    tasksWithStatus?.reduce<Record<string, string>>(
      (acc, task) => {
        acc[task.id] = task.status;
        return acc;
      },
      {},
    ) ?? {};

  const workerMap = uniqueWorkers.reduce<Record<string, { name: string; role: string | null }>>(
    (acc, worker) => {
      acc[worker.id] = { name: worker.name, role: worker.role };
      return acc;
    },
    {},
  );

  // Filtrer uniquement les rapports en attente
  type ReportType = {
    id: string;
    task_id: string | null;
    worker_id: string | null;
    photo_url: string | null;
    description: string | null;
    created_at: string;
  };

  const pendingReports: ReportType[] = [];

  allReports?.forEach((report) => {
    const taskStatus = taskStatusMap[report.task_id ?? ''];
    if (taskStatus !== 'done') {
      pendingReports.push(report);
    }
  });

  return (
    <AppShell
      heading="Rapports en attente"
      subheading="Tous les rapports en attente de validation."
      userEmail={user.email}
      primarySite={sites?.[0] ?? null}
    >
      <div className="mb-6">
        <Link
          href="/reports"
          className="inline-flex items-center gap-2 text-sm font-medium text-zinc-600 transition hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
        >
          <ArrowLeft className="h-4 w-4" />
          Retour aux rapports
        </Link>
      </div>

      <section className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
            Tous les rapports en attente
          </h2>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            {pendingReports.length > 0
              ? `${pendingReports.length} rapport${pendingReports.length > 1 ? 's' : ''} en attente de validation`
              : 'Aucun rapport en attente'}
          </p>
        </div>
        <PendingReportsList
          reports={pendingReports}
          taskMap={taskMap}
          workerMap={workerMap}
          siteMap={siteMap}
          totalCount={pendingReports.length}
        />
      </section>
    </AppShell>
  );
}

