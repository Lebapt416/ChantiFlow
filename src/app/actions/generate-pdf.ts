'use server';

import { createSupabaseServerClient } from '@/lib/supabase/server';
import { generateSiteSummary } from '@/lib/ai/summary';

export async function generatePDFAction(siteId: string) {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { error: 'Non authentifié' };
  }

  // Récupérer les données du chantier
  const [
    { data: site },
    { data: tasks },
    { data: workers },
    { data: reports },
  ] = await Promise.all([
    supabase.from('sites').select('*').eq('id', siteId).single(),
    supabase
      .from('tasks')
      .select('*')
      .eq('site_id', siteId)
      .order('created_at', { ascending: true }),
    supabase.from('workers').select('*').eq('site_id', siteId),
    supabase
      .from('reports')
      .select('*')
      .order('created_at', { ascending: false }),
  ]);

  if (!site || site.created_by !== user.id) {
    return { error: 'Chantier non trouvé ou accès refusé' };
  }

  // Générer le résumé IA
  let aiSummary: { summary: string; status: string } | null = null;
  if (tasks && tasks.length > 0) {
    try {
      aiSummary = await generateSiteSummary(site, tasks);
    } catch (error) {
      console.error('Erreur génération résumé IA pour PDF:', error);
    }
  }

  // Créer un map des workers pour faciliter l'accès
  const workerMap = new Map(
    workers?.map((w) => [w.id, { name: w.name, email: w.email, role: w.role }]) || [],
  );

  // Préparer les données pour le PDF
  const totalTasks = tasks?.length || 0;
  const doneTasks = tasks?.filter((t) => t.status === 'done').length || 0;
  const progress = totalTasks > 0 ? Math.round((doneTasks / totalTasks) * 100) : 0;

  return {
    success: true,
    data: {
      site: {
        name: site.name,
        deadline: site.deadline,
        created_at: site.created_at,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        address: (site as any).postal_code || 'Non spécifié',
      },
      stats: {
        totalTasks,
        doneTasks,
        pendingTasks: totalTasks - doneTasks,
        progress,
        workersCount: workers?.length || 0,
        reportsCount: reports?.length || 0,
      },
      tasks:
        tasks?.map((t) => {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const assignedWorker = (t as any).planned_worker_id
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ? workerMap.get((t as any).planned_worker_id)
            : null;
          return {
            title: t.title,
            status: t.status,
            role: t.required_role,
            duration_hours: t.duration_hours,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            startDate: (t as any).planned_start,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            endDate: (t as any).planned_end,
            assignedWorkerName: assignedWorker?.name || null,
          };
        }) || [],
      workers:
        workers?.map((w) => ({
          name: w.name,
          email: w.email,
          role: w.role,
        })) || [],
      aiSummary,
    },
  };
}

