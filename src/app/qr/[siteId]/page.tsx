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

        <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-6 dark:border-emerald-800 dark:bg-emerald-900/20">
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-white mb-2">
            Accès sécurisé
          </h2>
          <p className="text-sm text-zinc-600 dark:text-zinc-400 mb-4">
            Pour accéder à vos informations et envoyer des rapports, vous devez d'abord entrer votre code d'accès unique.
          </p>
          <a
            href={`/qr/${site.id}/verify`}
            className="inline-block rounded-lg bg-emerald-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-emerald-700"
          >
            Entrer mon code d'accès
          </a>
        </div>
      </div>
    </div>
  );
}

