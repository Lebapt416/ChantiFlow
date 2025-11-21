import { notFound } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { ReportForm } from './report-form';

type Params = {
  params: Promise<{
    siteId: string;
  }>;
};

export const dynamic = 'force-dynamic';

export default async function QrAccessPage({ params }: Params) {
  const { siteId } = await params;
  const supabase = await createSupabaseServerClient();

  const { data: site, error: siteError } = await supabase
    .from('sites')
    .select('id, name, deadline')
    .eq('id', siteId)
    .single();

  if (siteError || !site) {
    notFound();
  }

  // Récupérer le créateur du chantier pour chercher les workers au niveau du compte
  const { data: siteWithCreator } = await supabase
    .from('sites')
    .select('id, created_by')
    .eq('id', site.id)
    .single();

  // Récupérer les workers du chantier ET les workers au niveau du compte (réutilisables)
  const [tasksResult, siteWorkersResult, accountWorkersResult] = await Promise.all([
    supabase
      .from('tasks')
      .select('id, title, status, required_role')
      .eq('site_id', site.id)
      .order('created_at', { ascending: true }),
    supabase
      .from('workers')
      .select('name, email, role')
      .eq('site_id', site.id)
      .order('created_at', { ascending: true }),
    // Récupérer aussi les workers au niveau du compte si created_by existe
    siteWithCreator?.created_by
      ? supabase
          .from('workers')
          .select('name, email, role')
          .eq('created_by', siteWithCreator.created_by)
          .is('site_id', null)
          .order('created_at', { ascending: true })
      : Promise.resolve({ data: null }),
  ]);

  const tasks = tasksResult.data;
  const siteWorkers = siteWorkersResult.data ?? [];
  const accountWorkers = accountWorkersResult.data ?? [];

  // Combiner les workers du chantier et du compte (sans doublons par email)
  const workerMap = new Map<string, { name: string | null; email: string | null; role: string | null }>();
  
  // D'abord les workers du chantier
  siteWorkers.forEach((worker) => {
    if (worker.email) {
      workerMap.set(worker.email.toLowerCase(), worker);
    }
  });
  
  // Ensuite les workers du compte (ne remplacent pas ceux du chantier)
  accountWorkers.forEach((worker) => {
    if (worker.email && !workerMap.has(worker.email.toLowerCase())) {
      workerMap.set(worker.email.toLowerCase(), worker);
    }
  });

  const workers = Array.from(workerMap.values());

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-100 via-white to-zinc-50 p-6 text-zinc-900 dark:from-zinc-900 dark:via-zinc-800 dark:to-black dark:text-white">
      <div className="mx-auto max-w-4xl space-y-8">
        <header className="rounded-3xl border border-zinc-200 bg-white/80 p-8 backdrop-blur dark:border-white/10 dark:bg-white/5">
          <p className="text-xs uppercase tracking-[0.5em] text-zinc-500 dark:text-white/70">
            Accès chantier
          </p>
          <h1 className="mt-3 text-4xl font-semibold dark:text-white">
            {site.name}
          </h1>
          <p className="text-sm text-zinc-600 dark:text-white/70">
            Deadline{' '}
            {site.deadline
              ? new Date(site.deadline).toLocaleDateString('fr-FR')
              : 'Non communiquée'}
          </p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <ReportForm
            siteId={site.id}
            tasks={tasks?.map((task) => ({
              id: task.id,
              title: task.title,
            })) ?? []}
            workers={workers ?? []}
          />

          <div className="space-y-4 rounded-2xl border border-zinc-200 bg-white/80 p-6 backdrop-blur dark:border-white/10 dark:bg-white/5">
            <p className="text-xs uppercase tracking-[0.4em] text-zinc-500 dark:text-white/70">
              Tâches en cours
            </p>
            <div className="space-y-3">
              {tasks?.length ? (
                tasks.map((task) => (
                  <div
                    key={task.id}
                    className="rounded-xl border border-zinc-200 bg-white p-4 dark:border-white/10 dark:bg-white/10"
                  >
                    <p className="text-sm font-semibold dark:text-white">
                      {task.title}
                    </p>
                    <p className="text-xs text-zinc-500 dark:text-white/60">
                      {task.required_role || 'Tout rôle'}
                    </p>
                    <p className="text-xs text-zinc-500 dark:text-white/50">
                      {task.status === 'done' ? '✅ Terminé' : '⏳ À faire'}
                    </p>
                  </div>
                ))
              ) : (
                <p className="text-sm text-zinc-600 dark:text-white/60">
                  Aucune tâche disponible pour le moment. Contacte ton chef de
                  chantier.
                </p>
              )}
            </div>
            <div className="rounded-xl border border-zinc-200 bg-black/5 p-4 text-xs text-zinc-600 dark:border-white/10 dark:bg-black/30 dark:text-white/60">
              <p className="font-semibold text-zinc-900 dark:text-white">
                Comment ça marche ?
              </p>
              <ol className="mt-2 list-decimal space-y-1 pl-5">
                <li>Choisis ta tâche.</li>
                <li>Renseigne ton email pro et ton rôle.</li>
                <li>Ajoute une photo + ton rapport.</li>
                <li>Marque la tâche comme terminée si applicable.</li>
              </ol>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

