import Image from 'next/image';
import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { readWorkerSession } from '@/lib/worker-session';
import { WorkerNav } from '../components/worker-nav';

export const dynamic = 'force-dynamic';

type TimelineEntry = {
  id: string;
  title: string | null;
  date: string | null;
  status: string | null;
  siteName: string | null;
};

export default async function WorkerPlanningPage() {
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
    .select('id, title, status, planned_start, planned_end, created_at, site_id')
    .eq('assigned_worker_id', worker.id)
    .order('planned_start', { ascending: true });

  const siteIds = Array.from(new Set((tasks ?? []).map((task) => task.site_id).filter(Boolean))) as string[];
  let siteMap = new Map<string, { id: string; name: string | null }>();
  if (siteIds.length) {
    const { data: sitesData } = await supabase.from('sites').select('id, name').in('id', siteIds);
    siteMap = new Map((sitesData ?? []).map((site) => [site.id, site]));
  }

  const planning: TimelineEntry[] =
    tasks?.map((task) => ({
      id: task.id,
      title: task.title,
      date: task.planned_start || task.planned_end || task.created_at,
      status: task.status,
      siteName: task.site_id ? siteMap.get(task.site_id)?.name ?? 'Chantier' : 'Chantier',
    })) ?? [];

  planning.sort((a, b) => new Date(a.date ?? 0).getTime() - new Date(b.date ?? 0).getTime());

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
              <h1 className="text-2xl font-semibold text-zinc-900 dark:text-white">Planning global</h1>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Visualisez votre emploi du temps sur tous les chantiers.
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-8">
        {planning.length ? (
          <div className="space-y-6">
            <div className="rounded-3xl border border-zinc-200 bg-white/90 p-6 dark:border-zinc-800 dark:bg-zinc-900/90">
              <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">Timeline</p>
              <h2 className="text-2xl font-semibold text-zinc-900 dark:text-white">
                Vos {planning.length} prochaines missions
              </h2>
              <ol className="mt-6 space-y-4 border-l-2 border-dashed border-zinc-200 pl-6 dark:border-zinc-800">
                {planning.map((entry) => (
                  <li key={entry.id} className="relative pl-6">
                    <span className="absolute left-[-14px] top-1.5 h-3 w-3 rounded-full bg-emerald-500 ring-4 ring-emerald-100 dark:ring-emerald-900/40" />
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                        {entry.title || 'Tâche à venir'}
                      </p>
                      <span className="text-xs text-zinc-500 dark:text-zinc-400">{formatDate(entry.date)}</span>
                    </div>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">{entry.siteName}</p>
                    <span className="mt-1 inline-flex rounded-full bg-zinc-100 px-2 py-0.5 text-xs font-semibold text-zinc-600 dark:bg-zinc-800 dark:text-zinc-200">
                      {formatStatus(entry.status)}
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        ) : (
          <section className="rounded-3xl border border-dashed border-zinc-200 bg-white/80 p-6 text-center shadow-inner dark:border-zinc-800 dark:bg-zinc-950/40">
            <p className="text-lg font-semibold text-zinc-900 dark:text-white">Aucune mission planifiée</p>
            <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
              Rejoignez un chantier en scannant son QR code pour voir votre planning.
            </p>
          </section>
        )}
      </main>

      <WorkerNav />
    </div>
  );
}

function formatStatus(status?: string | null) {
  if (!status) return 'En attente';
  const normalized = status.toLowerCase();
  if (['done', 'completed', 'terminé', 'validated', 'valide', 'réalisé'].includes(normalized)) {
    return 'Terminé';
  }
  if (['in_progress', 'progress', 'en cours'].includes(normalized)) {
    return 'En cours';
  }
  if (['blocked', 'blocked_alert', 'bloqué'].includes(normalized)) {
    return 'Bloqué';
  }
  return status.charAt(0).toUpperCase() + status.slice(1);
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

