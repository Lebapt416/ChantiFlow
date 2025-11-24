'use server';

import { createSupabaseServerClient } from '@/lib/supabase/server';

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
        address: (site as any).address || 'Non spécifiée',
      },
      stats: {
        totalTasks,
        doneTasks,
        pendingTasks: totalTasks - doneTasks,
        progress,
        workersCount: workers?.length || 0,
        reportsCount: reports?.length || 0,
      },
      tasks: tasks?.map((t) => ({
        title: t.title,
        status: t.status,
        role: t.required_role,
        duration: t.duration_hours,
        startDate: (t as any).planned_start,
        endDate: (t as any).planned_end,
      })) || [],
      workers:
        workers?.map((w) => ({
          name: w.name,
          email: w.email,
          role: w.role,
        })) || [],
      reports:
        reports?.slice(0, 20).map((r) => ({
          description: r.description,
          created_at: r.created_at,
        })) || [],
    },
  };
}

