'use client';

import Link from 'next/link';
import { SimpleChart } from '@/components/simple-chart';

type Site = {
  id: string;
  name: string;
  deadline: string | null;
};

type Props = {
  sites: Site[];
  totalTasks: number;
  doneTasks: number;
  pendingTasks: number;
  nextDeadlines: Site[];
};

export function DashboardCharts({
  sites,
  totalTasks,
  doneTasks,
  pendingTasks,
  nextDeadlines,
}: Props) {
  // Données pour le graphique de progression
  const progressData = [
    { label: 'Terminées', value: doneTasks, color: '#10b981' },
    { label: 'En attente', value: pendingTasks, color: '#f59e0b' },
  ];

  // Données pour le graphique des chantiers
  const sitesData = sites.slice(0, 5).map((site) => ({
    label: site.name.length > 15 ? `${site.name.substring(0, 15)}...` : site.name,
    value: 1,
    color: '#3b82f6',
  }));

  return (
    <>
      <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
          Prochaines deadlines
        </h2>
        <ul className="mt-4 space-y-3">
          {nextDeadlines.length ? (
            nextDeadlines.map((site) => (
              <li
                key={site.id}
                className="rounded-2xl border border-zinc-200 px-4 py-3 dark:border-zinc-700"
              >
                <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                  {site.name}
                </p>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  {site.deadline
                    ? new Date(site.deadline).toLocaleDateString('fr-FR', {
                        day: '2-digit',
                        month: 'short',
                        year: 'numeric',
                      })
                    : 'Deadline à planifier'}
                </p>
              </li>
            ))
          ) : (
            <li className="rounded-2xl border border-dashed border-zinc-200 px-4 py-3 text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
              Pas encore de date prévue.
            </li>
          )}
        </ul>
      </div>

      <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="mb-4 text-lg font-semibold text-zinc-900 dark:text-white">
          Répartition des tâches
        </h2>
        <SimpleChart data={progressData} />
      </div>

      {sites.length > 0 && (
        <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Chantiers actifs
            </h2>
            <Link
              href="/sites"
              className="text-xs font-semibold text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
            >
              Voir tout →
            </Link>
          </div>
          <SimpleChart data={sitesData} />
        </div>
      )}
    </>
  );
}

