'use client';

import { useEffect, useState } from 'react';
import { Calendar, CheckSquare, FileText, Info } from 'lucide-react';
import { TasksTab } from '../[siteId]/tabs/tasks-tab';
import { PlanningTab } from '../[siteId]/tabs/planning-tab';
import { ReportsTab } from '../[siteId]/tabs/reports-tab';
import { InfoTab } from '../[siteId]/tabs/info-tab';

type Tab = 'tasks' | 'planning' | 'reports' | 'info';

type Props = {
  siteId: string;
};

export function WorkerSiteShell({ siteId }: Props) {
  const [activeTab, setActiveTab] = useState<Tab>('planning');
  const [workerId, setWorkerId] = useState<string | null>(null);
  const [workerName, setWorkerName] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!siteId) return;

    async function checkSession() {
      try {
        const response = await fetch('/api/worker/session', {
          method: 'GET',
          credentials: 'include',
        });

        if (response.ok) {
          const data = await response.json();
          if (data.workerId && (!data.siteId || data.siteId === siteId)) {
            setWorkerId(data.workerId);
            setWorkerName(data.name || '');
            setIsLoading(false);
            return;
          }
        }
      } catch (error) {
        console.error('Erreur vérification session:', error);
      }

      window.location.href = '/worker/login';
    }

    checkSession();
  }, [siteId]);

  if (isLoading || !workerId) {
    return (
      <div className="flex min-h-[320px] items-center justify-center rounded-2xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <div className="text-center">
          <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-zinc-300 border-t-emerald-600" />
          <p className="text-sm text-zinc-600 dark:text-zinc-400">Chargement...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-[520px] flex-col overflow-hidden rounded-3xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
      <div className="border-b border-zinc-200 bg-white px-4 py-4 dark:border-zinc-800 dark:bg-zinc-950">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">Votre chantier</h2>
        {workerName ? (
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            Connecté en tant que <strong>{workerName}</strong>
          </p>
        ) : null}
      </div>

      <div className="flex-1 overflow-y-auto">
        {activeTab === 'tasks' && <TasksTab siteId={siteId} workerId={workerId} />}
        {activeTab === 'planning' && <PlanningTab siteId={siteId} workerId={workerId} />}
        {activeTab === 'reports' && <ReportsTab siteId={siteId} workerId={workerId} />}
        {activeTab === 'info' && <InfoTab siteId={siteId} workerId={workerId} workerName={workerName} />}
      </div>

      <nav className="border-t border-zinc-200 bg-white text-xs font-medium dark:border-zinc-800 dark:bg-zinc-950">
        <div className="grid grid-cols-4">
          <button
            onClick={() => setActiveTab('tasks')}
            className={`flex flex-col items-center gap-1 py-3 transition ${
              activeTab === 'tasks' ? 'text-emerald-600 dark:text-emerald-400' : 'text-zinc-500 dark:text-zinc-400'
            }`}
          >
            <CheckSquare className="h-4 w-4" />
            Tâches
          </button>
          <button
            onClick={() => setActiveTab('planning')}
            className={`flex flex-col items-center gap-1 py-3 transition ${
              activeTab === 'planning'
                ? 'text-emerald-600 dark:text-emerald-400'
                : 'text-zinc-500 dark:text-zinc-400'
            }`}
          >
            <Calendar className="h-4 w-4" />
            Planning
          </button>
          <button
            onClick={() => setActiveTab('reports')}
            className={`flex flex-col items-center gap-1 py-3 transition ${
              activeTab === 'reports'
                ? 'text-emerald-600 dark:text-emerald-400'
                : 'text-zinc-500 dark:text-zinc-400'
            }`}
          >
            <FileText className="h-4 w-4" />
            Rapports
          </button>
          <button
            onClick={() => setActiveTab('info')}
            className={`flex flex-col items-center gap-1 py-3 transition ${
              activeTab === 'info' ? 'text-emerald-600 dark:text-emerald-400' : 'text-zinc-500 dark:text-zinc-400'
            }`}
          >
            <Info className="h-4 w-4" />
            Infos
          </button>
        </div>
      </nav>
    </div>
  );
}
