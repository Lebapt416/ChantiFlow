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
        <p className="text-sm text-ink-3">
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
            className="block rounded border border-rule-soft bg-paper p-4 transition hover:bg-paper-2 dark:border-rule dark:bg-ink dark:hover:bg-paper-2/10"
          >
            <div className="flex items-center justify-between gap-3">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-orange dark:text-green flex-shrink-0" />
                  <p className="text-sm font-semibold text-ink dark:text-paper truncate">
                    {task?.title ?? 'Tâche inconnue'}
                  </p>
                </div>
                <p className="text-xs text-ink-3 truncate mt-1">
                  {siteName} • {worker?.name ?? 'Employé inconnu'}
                </p>
                <p className="text-xs text-ink-3 mt-1">
                  {new Date(report.created_at).toLocaleDateString('fr-FR', {
                    day: '2-digit',
                    month: 'short',
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </p>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <Eye className="h-4 w-4 text-ink-3" />
              </div>
            </div>
          </Link>
        );
      })}

      {totalCount > 10 && (
        <Link
          href="/reports/validated"
          className="flex items-center justify-center gap-2 border border-rule-soft bg-paper px-4 py-3 font-mono text-[10px] uppercase tracking-widest text-ink-2 transition hover:bg-paper-2 hover:text-ink dark:border-rule dark:bg-ink"
        >
          <span>Voir tous les rapports validés</span>
          <ArrowRight className="h-4 w-4" />
        </Link>
      )}
    </div>
  );
}
