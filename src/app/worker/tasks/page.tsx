import Image from 'next/image';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { readWorkerSession } from '@/lib/worker-session';
import { WorkerNav } from '../components/worker-nav';
import { WorkerTasksList, type WorkerTaskGroup } from '../components/worker-tasks-list';

export const dynamic = 'force-dynamic';

const DONE = ['done', 'completed', 'terminé', 'validated', 'valide', 'réalisé'];
const PROGRESS = ['in_progress', 'progress', 'en cours', 'ongoing', 'running'];

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

  const uniqueTasks = Array.from(new Map((tasks ?? []).map((task) => [task.id, task])).values());

  const siteIds = uniqueTasks.reduce<string[]>((acc, task) => {
    if (task.site_id && !acc.includes(task.site_id)) {
      acc.push(task.site_id);
    }
    return acc;
  }, []);

  let siteMap = new Map<string, { id: string; name: string | null }>();
  if (siteIds.length) {
    const { data: siteData } = await supabase.from('sites').select('id, name').in('id', siteIds);
    siteMap = new Map((siteData ?? []).map((site) => [site.id, site]));
  }

  const tasksWithMeta = uniqueTasks.map((task) => {
    const normalized = (task.status ?? '').toLowerCase();
    const group: WorkerTaskGroup['key'] = DONE.includes(normalized)
      ? 'done'
      : PROGRESS.includes(normalized)
        ? 'progress'
        : 'todo';
    return {
      id: task.id,
      title: task.title,
      status: task.status,
      required_role: task.required_role,
      siteId: task.site_id ?? null,
      siteName: task.site_id ? siteMap.get(task.site_id)?.name ?? 'Chantier' : 'Chantier',
      date: task.planned_start || task.planned_end || task.created_at,
      group,
    };
  });

  const groups: WorkerTaskGroup[] = [
    {
      key: 'todo',
      label: 'À faire',
      description: 'Tâches à démarrer',
      accent: 'text-amber-500',
      badge: 'bg-amber-100 text-amber-800',
    },
    {
      key: 'progress',
      label: 'En cours',
      description: 'Missions déjà lancées',
      accent: 'text-sky-500',
      badge: 'bg-sky-100 text-sky-800',
    },
    {
      key: 'done',
      label: 'Terminées',
      description: 'Validées ou livrées',
      accent: 'text-green',
      badge: 'bg-paper-2 text-ink',
    },
  ];

  return (
    <div className="min-h-screen bg-paper pb-32 dark:bg-ink">
      <header className="border-b border-rule-soft bg-paper px-4 py-6 dark:border-rule dark:bg-ink">
        <div className="mx-auto flex max-w-5xl flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded bg-ink dark:bg-paper">
              <Image src="/logo.svg" alt="ChantiFlow" width={32} height={32} priority className="h-8 w-8" />
            </div>
            <div>
              <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3">ChantiFlow</p>
              <h1 className="font-serif text-[24px] text-ink dark:text-paper">Mes tâches</h1>
              <p className="text-sm text-ink-3">
                Retrouvez l’ensemble de vos missions ({tasksWithMeta.length}) et validez vos actions en direct.
              </p>
            </div>
          </div>
          <Link
            href="/worker/scanner"
            className="inline-flex items-center gap-2 border border-orange px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-orange transition hover:bg-paper-2"
          >
            Scanner un chantier
          </Link>
        </div>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-8">
        <WorkerTasksList tasks={tasksWithMeta} groups={groups} />
      </main>

      <WorkerNav />
    </div>
  );
}
