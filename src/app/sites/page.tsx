import Link from 'next/link';
import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { FolderKanban, Plus } from 'lucide-react';

export const metadata = {
  title: 'Chantiers | ChantiFlow',
};

export default async function SitesPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: sites } = await supabase
    .from('sites')
    .select('id, name, deadline, created_at')
    .eq('created_by', user.id)
    .order('created_at', { ascending: false });

  // Récupérer les stats pour chaque chantier
  const siteIds = sites?.map((site) => site.id) ?? [];
  const siteStats: Record<string, { tasks: number; done: number; workers: number }> = {};

  if (siteIds.length > 0) {
    const [{ data: tasks }, { data: workers }] = await Promise.all([
      supabase
        .from('tasks')
        .select('site_id, status')
        .in('site_id', siteIds),
      supabase
        .from('workers')
        .select('site_id')
        .in('site_id', siteIds),
    ]);

    tasks?.forEach((task) => {
      if (!siteStats[task.site_id]) {
        siteStats[task.site_id] = { tasks: 0, done: 0, workers: 0 };
      }
      siteStats[task.site_id].tasks++;
      if (task.status === 'done') {
        siteStats[task.site_id].done++;
      }
    });

    workers?.forEach((worker) => {
      if (!siteStats[worker.site_id]) {
        siteStats[worker.site_id] = { tasks: 0, done: 0, workers: 0 };
      }
      siteStats[worker.site_id].workers++;
    });
  }

  return (
    <AppShell
      heading="Chantiers"
      subheading="Tous vos chantiers en un coup d'œil"
      userEmail={user.email}
    >
      <div className="overflow-x-auto pb-4">
        <div className="flex gap-4 min-w-max">
          {sites?.map((site) => {
            const stats = siteStats[site.id] || { tasks: 0, done: 0, workers: 0 };
            const progress = stats.tasks > 0 ? Math.round((stats.done / stats.tasks) * 100) : 0;
            return (
              <Link
                key={site.id}
                href={`/site/${site.id}`}
                className="flex-shrink-0 w-80 rounded-2xl border border-zinc-200 bg-white p-6 shadow-lg shadow-black/5 transition hover:shadow-xl hover:scale-105 dark:border-zinc-800 dark:bg-zinc-900"
              >
                <div className="mb-4 flex items-start justify-between">
                  <div className="flex-1">
                    <div className="mb-2 flex items-center gap-2">
                      <FolderKanban className="h-5 w-5 text-zinc-500 dark:text-zinc-400" />
                      <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
                        {site.name}
                      </h3>
                    </div>
                    {site.deadline ? (
                      <p className="text-xs text-zinc-500 dark:text-zinc-400">
                        Deadline : {new Date(site.deadline).toLocaleDateString('fr-FR')}
                      </p>
                    ) : (
                      <p className="text-xs text-zinc-500 dark:text-zinc-400">
                        Deadline non définie
                      </p>
                    )}
                  </div>
                </div>

                <div className="mb-4 grid grid-cols-3 gap-3">
                  <div className="rounded-lg border border-zinc-100 bg-zinc-50 p-2 text-center dark:border-zinc-800 dark:bg-zinc-800">
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">Tâches</p>
                    <p className="mt-1 text-lg font-semibold text-zinc-900 dark:text-white">
                      {stats.tasks}
                    </p>
                  </div>
                  <div className="rounded-lg border border-zinc-100 bg-zinc-50 p-2 text-center dark:border-zinc-800 dark:bg-zinc-800">
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">Terminées</p>
                    <p className="mt-1 text-lg font-semibold text-emerald-600 dark:text-emerald-400">
                      {stats.done}
                    </p>
                  </div>
                  <div className="rounded-lg border border-zinc-100 bg-zinc-50 p-2 text-center dark:border-zinc-800 dark:bg-zinc-800">
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">Équipe</p>
                    <p className="mt-1 text-lg font-semibold text-zinc-900 dark:text-white">
                      {stats.workers}
                    </p>
                  </div>
                </div>

                <div className="mb-4">
                  <div className="mb-1 flex items-center justify-between text-xs">
                    <span className="text-zinc-500 dark:text-zinc-400">Progression</span>
                    <span className="font-semibold text-zinc-900 dark:text-white">{progress}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-zinc-200 dark:bg-zinc-800">
                    <div
                      className="h-full rounded-full bg-emerald-500 transition-all"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>

                <div className="mt-4 rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-center text-xs font-semibold text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700">
                  Ouvrir le chantier →
                </div>
              </Link>
            );
          })}

          {/* Carte pour créer un nouveau chantier */}
          <Link
            href="/dashboard"
            className="flex-shrink-0 w-80 rounded-2xl border-2 border-dashed border-zinc-300 bg-zinc-50 p-6 shadow-lg shadow-black/5 transition hover:border-zinc-400 hover:bg-zinc-100 hover:shadow-xl hover:scale-105 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:border-zinc-600"
          >
            <div className="flex h-full flex-col items-center justify-center text-center">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full border-2 border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-800">
                <Plus className="h-8 w-8 text-zinc-500 dark:text-zinc-400" />
              </div>
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
                Nouveau chantier
              </h3>
              <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
                Créez un nouveau projet de construction
              </p>
            </div>
          </Link>
        </div>
      </div>
    </AppShell>
  );
}

