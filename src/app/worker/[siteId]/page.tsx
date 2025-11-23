'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { TasksTab } from './tabs/tasks-tab';
import { ReportsTab } from './tabs/reports-tab';
import { InfoTab } from './tabs/info-tab';
import { PlanningTab } from './tabs/planning-tab';
import { CheckSquare, FileText, Info, Calendar } from 'lucide-react';

type Tab = 'tasks' | 'reports' | 'info' | 'planning';

export default function WorkerMobilePage() {
  const params = useParams();
  const siteId = params?.siteId as string;
  const [activeTab, setActiveTab] = useState<Tab>('planning');
  const [workerId, setWorkerId] = useState<string | null>(null);
  const [workerName, setWorkerName] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!siteId) return;

    // Vérifier la session worker via les cookies (gérés côté serveur)
    // Si pas de session, rediriger vers la page de connexion
    async function checkSession() {
      try {
        const response = await fetch('/api/worker/session', {
          method: 'GET',
          credentials: 'include',
        });

        if (response.ok) {
          const data = await response.json();
          if (data.workerId && data.siteId === siteId) {
            setWorkerId(data.workerId);
            setWorkerName(data.name || '');
            setIsLoading(false);
            return;
          }
        }
      } catch (error) {
        console.error('Erreur vérification session:', error);
      }

      // Pas de session valide, rediriger vers la connexion
      window.location.href = `/worker/login?siteId=${siteId}`;
    }

    checkSession();
  }, [siteId]);

  if (isLoading || !workerId || !siteId) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-zinc-50 dark:bg-zinc-900">
        <div className="text-center">
          <div className="mb-4 h-8 w-8 animate-spin rounded-full border-4 border-zinc-300 border-t-emerald-600 mx-auto"></div>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">Chargement...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 dark:bg-zinc-900 pb-16 sm:pb-20">
      {/* Header */}
      <header className="border-b border-zinc-200 bg-white px-3 sm:px-4 py-3 sm:py-4 dark:border-zinc-800 dark:bg-zinc-950">
        <h1 className="text-lg sm:text-xl font-bold text-zinc-900 dark:text-white">
          ChantiFlow
        </h1>
        {workerName && (
          <p className="text-xs sm:text-sm text-zinc-600 dark:text-zinc-400">
            {workerName}
          </p>
        )}
      </header>

      {/* Content */}
      <main className="flex-1 overflow-y-auto">
        {activeTab === 'tasks' && <TasksTab siteId={siteId} workerId={workerId} />}
        {activeTab === 'planning' && <PlanningTab siteId={siteId} workerId={workerId} />}
        {activeTab === 'reports' && <ReportsTab siteId={siteId} workerId={workerId} />}
        {activeTab === 'info' && <InfoTab siteId={siteId} workerId={workerId} workerName={workerName} />}
      </main>

      {/* Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 right-0 border-t border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950 z-50">
        <div className="grid grid-cols-4">
          <button
            onClick={() => setActiveTab('tasks')}
            className={`flex flex-col items-center gap-1 py-2 sm:py-3 transition ${
              activeTab === 'tasks'
                ? 'text-emerald-600 dark:text-emerald-400'
                : 'text-zinc-500 dark:text-zinc-400'
            }`}
          >
            <CheckSquare className="h-4 w-4 sm:h-5 sm:w-5" />
            <span className="text-[10px] sm:text-xs font-medium">Tâches</span>
          </button>
          <button
            onClick={() => setActiveTab('planning')}
            className={`flex flex-col items-center gap-1 py-2 sm:py-3 transition ${
              activeTab === 'planning'
                ? 'text-emerald-600 dark:text-emerald-400'
                : 'text-zinc-500 dark:text-zinc-400'
            }`}
          >
            <Calendar className="h-4 w-4 sm:h-5 sm:w-5" />
            <span className="text-[10px] sm:text-xs font-medium">Planning</span>
          </button>
          <button
            onClick={() => setActiveTab('reports')}
            className={`flex flex-col items-center gap-1 py-2 sm:py-3 transition ${
              activeTab === 'reports'
                ? 'text-emerald-600 dark:text-emerald-400'
                : 'text-zinc-500 dark:text-zinc-400'
            }`}
          >
            <FileText className="h-4 w-4 sm:h-5 sm:w-5" />
            <span className="text-[10px] sm:text-xs font-medium">Rapports</span>
          </button>
          <button
            onClick={() => setActiveTab('info')}
            className={`flex flex-col items-center gap-1 py-2 sm:py-3 transition ${
              activeTab === 'info'
                ? 'text-emerald-600 dark:text-emerald-400'
                : 'text-zinc-500 dark:text-zinc-400'
            }`}
          >
            <Info className="h-4 w-4 sm:h-5 sm:w-5" />
            <span className="text-[10px] sm:text-xs font-medium">Infos</span>
          </button>
        </div>
      </nav>
    </div>
  );
}

