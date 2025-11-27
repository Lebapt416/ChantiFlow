'use client';

import { useState, useEffect } from 'react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { User, Mail, Briefcase, Calendar } from 'lucide-react';

type Props = {
  siteId: string;
  workerId: string;
  workerName: string;
};

export function InfoTab({ siteId, workerId, workerName }: Props) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [worker, setWorker] = useState<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [site, setSite] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      const supabase = createSupabaseBrowserClient();

      // Charger les infos du worker
      const { data: workerData } = await supabase
        .from('workers')
        .select('name, email, role, created_at')
        .eq('id', workerId)
        .single();

      if (workerData) {
        setWorker(workerData);
      }

      // Charger les infos du chantier
      const { data: siteData } = await supabase
        .from('sites')
        .select('name, deadline')
        .eq('id', siteId)
        .single();

      if (siteData) {
        setSite(siteData);
      }

      setIsLoading(false);
    }

    loadData();
  }, [siteId, workerId]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-zinc-300 border-t-emerald-600"></div>
      </div>
    );
  }

  return (
    <div className="p-3 sm:p-4 md:p-6 space-y-4 sm:space-y-6">
      <div>
        <h2 className="text-base sm:text-lg font-semibold text-zinc-900 dark:text-white mb-3 sm:mb-4">
          Mes informations
        </h2>

        <div className="space-y-2 sm:space-y-3">
          <div className="rounded-lg border border-zinc-200 bg-white p-3 sm:p-4 dark:border-zinc-800 dark:bg-zinc-900">
              <div className="flex items-center gap-2 sm:gap-3">
                <User className="h-4 w-4 sm:h-5 sm:w-5 text-zinc-400 flex-shrink-0" />
                <div className="min-w-0">
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">Nom</p>
                  <p className="font-medium text-sm sm:text-base text-zinc-900 dark:text-white break-words">
                    {worker?.name || workerName}
                  </p>
                </div>
              </div>
          </div>

          {worker?.email && (
            <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
              <div className="flex items-center gap-2 sm:gap-3">
                <Mail className="h-4 w-4 sm:h-5 sm:w-5 text-zinc-400 flex-shrink-0" />
                <div className="min-w-0">
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">Email</p>
                  <p className="font-medium text-sm sm:text-base text-zinc-900 dark:text-white break-all">
                    {worker.email}
                  </p>
                </div>
              </div>
            </div>
          )}

          {worker?.role && (
            <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
              <div className="flex items-center gap-3">
                <Briefcase className="h-5 w-5 text-zinc-400" />
                <div>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">Rôle</p>
                  <p className="font-medium text-zinc-900 dark:text-white">
                    {worker.role}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {site && (
        <div>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white mb-4">
            Chantier
          </h2>

          <div className="space-y-3">
            <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
              <div className="flex items-center gap-3">
                <Briefcase className="h-5 w-5 text-zinc-400" />
                <div>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">Nom du chantier</p>
                  <p className="font-medium text-zinc-900 dark:text-white">
                    {site.name}
                  </p>
                </div>
              </div>
            </div>

            {site.deadline && (
              <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
                <div className="flex items-center gap-3">
                  <Calendar className="h-5 w-5 text-zinc-400" />
                  <div>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">Deadline</p>
                    <p className="font-medium text-zinc-900 dark:text-white">
                      {new Date(site.deadline).toLocaleDateString('fr-FR')}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <p className="text-xs text-zinc-500 dark:text-zinc-400 mb-2">
          Besoin d&apos;aide ?
        </p>
        <p className="text-sm text-zinc-700 dark:text-zinc-300">
          Contactez votre chef de chantier pour toute question.
        </p>
      </div>
    </div>
  );
}

