'use client';

import Link from 'next/link';
import { CheckCircle2, Eye, ArrowRight } from 'lucide-react';

type Report = {
  id: string;
  task_id: string | null;
  worker_id: string | null;
  photo_url: string | null;
  description: string | null;
  created_at: string;
};

type Task = {
  title: string;
  site_id: string;
};

type Worker = {
  name: string;
  role: string | null;
};

type Props = {
  reports: Report[];
  taskMap: Record<string, Task>;
  workerMap: Record<string, Worker>;
  siteMap: Record<string, string>;
  totalCount: number;
};

export function ValidatedReportsList({
  reports,
  taskMap,
  workerMap,
  siteMap,
  totalCount,
}: Props) {
  if (reports.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Aucun rapport validé pour le moment.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {reports.map((report) => {
        const task = taskMap[report.task_id ?? ''];
        const worker = workerMap[report.worker_id ?? ''];
        const siteName = siteMap[task?.site_id ?? ''] ?? 'Site inconnu';

        return (
          <Link
            key={report.id}
            href={`/reports/${report.id}`}
            className="block rounded-lg border border-zinc-200 bg-white p-4 transition hover:border-zinc-300 hover:shadow-sm dark:border-zinc-700 dark:bg-zinc-900 dark:hover:border-zinc-600"
          >
            <div className="flex items-center justify-between gap-3">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
                  <p className="text-sm font-semibold text-zinc-900 dark:text-white truncate">
                    {task?.title ?? 'Tâche inconnue'}
                  </p>
                </div>
                <p className="text-xs text-zinc-500 dark:text-zinc-400 truncate mt-1">
                  {siteName} • {worker?.name ?? 'Employé inconnu'}
                </p>
                <p className="text-xs text-zinc-400 dark:text-zinc-500 mt-1">
                  {new Date(report.created_at).toLocaleDateString('fr-FR', {
                    day: '2-digit',
                    month: 'short',
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </p>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <Eye className="h-4 w-4 text-zinc-400 dark:text-zinc-500" />
              </div>
            </div>
          </Link>
        );
      })}

      {totalCount > 10 && (
        <Link
          href="/reports/validated"
          className="flex items-center justify-center gap-2 rounded-lg border border-zinc-200 bg-zinc-50 px-4 py-3 text-sm font-semibold text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
        >
          <span>Voir tous les rapports validés</span>
          <ArrowRight className="h-4 w-4" />
        </Link>
      )}
    </div>
  );
}

