'use client';

import { useState } from 'react';
import Image from 'next/image';
import { CheckCircle2 } from 'lucide-react';
import { ReportDetailModal } from './report-detail-modal';

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
  taskStatusMap: Record<string, string>;
};

export function ReportsList({
  reports,
  taskMap,
  workerMap,
  siteMap,
  taskStatusMap,
}: Props) {
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);

  if (reports.length === 0) {
    return (
      <p className="text-sm text-zinc-500 dark:text-zinc-400">
        Aucun rapport pour le moment.
      </p>
    );
  }

  return (
    <>
      <div className="space-y-4">
        {reports.map((report) => {
          const task = taskMap[report.task_id ?? ''];
          const worker = workerMap[report.worker_id ?? ''];
          const siteName = siteMap[task?.site_id ?? ''] ?? 'Site inconnu';
          const taskStatus = taskStatusMap[report.task_id ?? ''];
          const isTaskDone = taskStatus === 'done';

          return (
            <button
              key={report.id}
              onClick={() => setSelectedReport(report)}
              className="w-full text-left rounded-2xl border border-zinc-200 p-4 transition hover:border-zinc-300 hover:shadow-md dark:border-zinc-700 dark:hover:border-zinc-600"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                      {task?.title ?? 'Tâche inconnue'}
                    </p>
                    {isTaskDone && (
                      <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
                    )}
                  </div>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                    {siteName}
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
              {report.description && (
                <p className="mt-3 text-sm text-zinc-700 dark:text-zinc-200 line-clamp-2">
                  {report.description}
                </p>
              )}
              {report.photo_url && (
                <div className="mt-3 overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-700">
                  <Image
                    src={report.photo_url}
                    alt="Photo rapport"
                    width={800}
                    height={320}
                    className="h-48 w-full object-cover"
                  />
                </div>
              )}
              {isTaskDone && (
                <div className="mt-3 flex items-center gap-2 text-xs text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 className="h-4 w-4" />
                  <span className="font-medium">Tâche terminée</span>
                </div>
              )}
            </button>
          );
        })}
      </div>

      {selectedReport && (
        <ReportDetailModal
          report={selectedReport}
          task={
            selectedReport.task_id
              ? {
                  id: selectedReport.task_id,
                  title: taskMap[selectedReport.task_id]?.title ?? 'Tâche inconnue',
                  site_id: taskMap[selectedReport.task_id]?.site_id ?? '',
                  status: taskStatusMap[selectedReport.task_id] ?? 'pending',
                }
              : null
          }
          worker={
            selectedReport.worker_id
              ? workerMap[selectedReport.worker_id] ?? null
              : null
          }
          siteName={
            selectedReport.task_id
              ? siteMap[taskMap[selectedReport.task_id]?.site_id ?? ''] ?? 'Site inconnu'
              : 'Site inconnu'
          }
          isOpen={!!selectedReport}
          onClose={() => setSelectedReport(null)}
        />
      )}
    </>
  );
}

