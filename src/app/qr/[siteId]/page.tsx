import { notFound, redirect } from 'next/navigation';
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

  // Rediriger automatiquement vers la page de connexion worker
  redirect(`/worker/login?siteId=${site.id}`);
}

