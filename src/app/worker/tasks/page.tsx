import Image from 'next/image';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { readWorkerSession } from '@/lib/worker-session';
import { WorkerNav } from '../components/worker-nav';

export const dynamic = 'force-dynamic';

const doneStatuses = new Set(['done', 'completed', 'terminé', 'validated', 'valide', 'réalisé']);
const progressStatuses = new Set(['in_progress', 'progress', 'en cours', 'ongoing', 'running']);

export default async function WorkerTasksPage() {
  const session = await readWorkerSession();
  if (!session?.workerId) {
    redirect('/worker/login');
  }

  const supabase = await createSupabaseServerClient();
  const { data: worker } = await supabase
    .from('workers')
    .select('id, name')
    .eq('id', session.workerId)
    .single();

  if (!worker) {
    redirect('/worker/login');
  }

  const { data: tasks } = await supabase
    .from('tasks')
    .select('id, title, status, required_role, planned_start, planned_end, created_at, site_id')
    .eq('assigned_worker_id', worker.id)
    .order('planned_start', { ascending: true });

  const siteIds =
    tasks?.reduce<string[]>((acc, task) => {
      if (task.site_id && !acc.includes(task.site_id)) {
        acc.push(task.site_id);
      }
      return acc;
    }, []) ?? [];

  let siteMap = new Map<string, { id: string; name: string | null }>();
  if (siteIds.length) {
    const { data: siteData } = await supabase.from('sites').select('id, name').in('id', siteIds);
    siteMap = new Map((siteData ?? []).map((site) => [site.id, site]));
  }

  const tasksWithMeta =
    tasks?.map((task) => {
      const normalized = (task.status ?? '').toLowerCase();
      const group = doneStatuses.has(normalized) ? 'done' : progressStatuses.has(normalized) ? 'progress' : 'todo';
      return {
        ...task,
        group,
        siteName: task.site_id ? siteMap.get(task.site_id)?.name ?? 'Chantier' : 'Chantier',
        date: task.planned_start || task.planned_end || task.created_at,
      };
    }) ?? [];

  const groups = [
    { key: 'todo', label: 'À faire', description: 'Tâches à démarrer', accent: 'text-amber-600', pill: 'bg-amber-100 text-amber-800' },
    { key: 'progress', label: 'En cours', description: 'Missions déjà lancées', accent: 'text-sky-600', pill: 'bg-sky-100 text-sky-800' },
    { key: 'done', label: 'Terminées', description: 'Validées ou livrées', accent: 'text-emerald-600', pill: 'bg-emerald-100 text-emerald-800' },
  ] as const;

  const tasksByGroup = Object.fromEntries(groups.map((group) => [group.key, tasksWithMeta.filter((task) => task.group === group.key)]));
  const totalTasks = tasksWithMeta.length;

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-50 via-white to-zinc-100 pb-32 dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950">
      <header className="border-b border-white/80 bg-white/90 px-4 py-6 backdrop-blur dark:border-zinc-900/60 dark:bg-zinc-900/80">
        <div className="mx-auto flex max-w-5xl flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-zinc-900 shadow-lg shadow-black/10 dark:bg-white">
              <Image src="/logo.svg" alt="ChantiFlow" width={32} height={32} priority className="h-8 w-8" />
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-zinc-500 dark:text-zinc-400">ChantiFlow</p>
              <h1 className="text-2xl font-semibold text-zinc-900 dark:text-white">Mes tâches</h1>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Retrouvez l’ensemble de vos missions ({totalTasks}) et validez vos actions en direct.
              </p>
            </div>
          </div>
          <Link
            href="/worker/scanner"
            className="inline-flex items-center gap-2 rounded-full bg-emerald-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-emerald-700"
          >
            Scanner un chantier
          </Link>
        </div>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-8">
        <section className="grid gap-4 md:grid-cols-3">
          {groups.map((group) => (
            <div
              key={group.key}
              className="rounded-3xl border border-zinc-200 bg-white/90 p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900/90"
            >
              <p className={`text-xs uppercase tracking-[0.3em] ${group.accent}`}>{group.label}</p>
              <p className="mt-2 text-3xl font-semibold text-zinc-900 dark:text-white">{tasksByGroup[group.key].length}</p>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">{group.description}</p>
            </div>
          ))}
        </section>

        {groups.map((group) => (
          <section
            key={group.key}
            className="rounded-3xl border border-zinc-200 bg-white/90 p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900/90"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-xs uppercase tracking-[0.3em] ${group.accent}`}>{group.label}</p>
                <h2 className="text-2xl font-semibold text-zinc-900 dark:text-white">
                  {tasksByGroup[group.key].length} mission(s)
                </h2>
              </div>
              <span className={`rounded-full px-3 py-1 text-xs font-semibold ${group.pill}`}>
                {group.description}
              </span>
            </div>
            <div className="mt-6 space-y-4">
              {tasksByGroup[group.key].length ? (
                tasksByGroup[group.key].map((task) => (
                  <div
                    key={task.id}
                    className="flex flex-col gap-2 rounded-2xl border border-zinc-200 bg-zinc-50/70 px-4 py-3 dark:border-zinc-700 dark:bg-zinc-900/60"
                  >
                    <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
                      <span>{task.siteName}</span>
                      <span>{task.date ? formatDate(task.date) : 'À planifier'}</span>
                    </div>
                    <p className="text-sm font-semibold text-zinc-900 dark:text-white">{task.title || 'Tâche'}</p>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      {task.required_role ? `Rôle : ${task.required_role}` : 'Rôle non précisé'}
                    </p>
                    {task.site_id ? (
                      <Link
                        href={`/worker/${task.site_id}`}
                        className="inline-flex w-full items-center justify-center rounded-xl bg-emerald-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-emerald-700"
                      >
                        Ouvrir la fiche chantier
                      </Link>
                    ) : null}
                  </div>
                ))
              ) : (
                <p className="text-sm text-zinc-500 dark:text-zinc-400">Aucune tâche dans cette catégorie.</p>
              )}
            </div>
          </section>
        ))}

        {!tasksWithMeta.length && (
          <section className="rounded-3xl border border-dashed border-zinc-200 bg-white/80 p-6 text-center shadow-inner dark:border-zinc-800 dark:bg-zinc-950/40">
            <p className="text-lg font-semibold text-zinc-900 dark:text-white">Aucune mission assignée</p>
            <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
              Scannez un chantier pour recevoir vos premières tâches.
            </p>
          </section>
        )}
      </main>

      <WorkerNav />
    </div>
  );
}

function formatDate(date?: string | null) {
  if (!date) return 'Non définie';
  try {
    return new Date(date).toLocaleDateString('fr-FR', {
      day: '2-digit',
      month: 'short',
    });
  } catch {
    return 'Non définie';
  }
}

