import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';
import { AnalyticsDashboard } from '@/components/analytics-dashboard';

export const metadata = {
  title: 'Analytics | ChantiFlow',
  description: 'Tableau de bord analytics en temps réel',
};

export const dynamic = 'force-dynamic';

export default async function AnalyticsPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  // Vérifier que l'utilisateur est connecté et qu'il s'agit du compte autorisé
  if (!user || user.email !== 'bcb83@icloud.com') {
    redirect('/login');
  }

  // Utiliser le client admin pour récupérer tous les utilisateurs
  const adminClient = createSupabaseAdminClient();

  // Récupérer toutes les données du site
  let allUsers: any[] = [];
  let allSites: any[] = [];
  let allTasks: any[] = [];
  let allReports: any[] = [];
  let allWorkers: any[] = [];

  try {
    const [
      { data: allUsersData },
      { data: sitesData },
      { data: tasksData },
      { data: reportsData },
      { data: workersData },
    ] = await Promise.all([
      // Tous les utilisateurs via admin client
      adminClient.auth.admin.listUsers(),
      // Tous les chantiers
      adminClient.from('sites').select('id, name, deadline, created_at, completed_at, created_by'),
      // Toutes les tâches
      adminClient.from('tasks').select('id, site_id, status, created_at, required_role, duration_hours'),
      // Tous les rapports
      adminClient.from('reports').select('id, task_id, worker_id, created_at, photo_url'),
      // Tous les workers
      adminClient.from('workers').select('id, site_id, name, email, role, status, created_at'),
    ]);

    allUsers = allUsersData?.users ?? [];
    allSites = sitesData ?? [];
    allTasks = tasksData ?? [];
    allReports = reportsData ?? [];
    allWorkers = workersData ?? [];
  } catch (error) {
    console.error('Erreur lors de la récupération des données analytics:', error);
    // Continuer avec des données vides plutôt que de planter
  }

  // Calculer les statistiques
  const totalUsers = allUsers.length;
  const totalSites = allSites.length;
  const activeSites = allSites.filter((site) => !site.completed_at).length;
  const completedSites = allSites.filter((site) => site.completed_at).length;
  const totalTasks = allTasks.length;
  const doneTasks = allTasks.filter((task) => task.status === 'done').length;
  const pendingTasks = totalTasks - doneTasks;
  const totalReports = allReports.length;
  const totalWorkers = allWorkers.length;
  const approvedWorkers = allWorkers.filter((w) => w.status === 'approved' || !w.status).length;
  const pendingWorkers = allWorkers.filter((w) => w.status === 'pending').length;

  // Statistiques par date (30 derniers jours)
  const last30Days = Array.from({ length: 30 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (29 - i));
    return date.toISOString().split('T')[0];
  });

  // Sites créés par jour
  const sitesByDay = last30Days.map((day) => {
    const count = allSites.filter((site) => {
      const siteDate = new Date(site.created_at).toISOString().split('T')[0];
      return siteDate === day;
    }).length;
    return { date: day, sites: count };
  });

  // Tâches créées par jour
  const tasksByDay = last30Days.map((day) => {
    const count = allTasks.filter((task) => {
      const taskDate = new Date(task.created_at).toISOString().split('T')[0];
      return taskDate === day;
    }).length;
    return { date: day, tasks: count };
  });

  // Rapports créés par jour
  const reportsByDay = last30Days.map((day) => {
    const count = allReports.filter((report) => {
      const reportDate = new Date(report.created_at).toISOString().split('T')[0];
      return reportDate === day;
    }).length;
    return { date: day, reports: count };
  });

  // Utilisateurs créés par jour
  const usersByDay = last30Days.map((day) => {
    const count = allUsers.filter((user) => {
      const userDate = new Date(user.created_at).toISOString().split('T')[0];
      return userDate === day;
    }).length;
    return { date: day, users: count };
  });

  // Répartition des rôles
  const rolesDistribution = allWorkers.reduce((acc, worker) => {
    const role = worker.role || 'Non spécifié';
    acc[role] = (acc[role] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  // Répartition des statuts de tâches
  const taskStatusDistribution = {
    'Terminées': doneTasks,
    'En attente': pendingTasks,
  };

  // Top 10 chantiers par nombre de tâches
  const sitesByTasks = allSites
    .map((site) => {
      const taskCount = allTasks.filter((task) => task.site_id === site.id).length;
      return {
        name: site.name,
        tasks: taskCount,
      };
    })
    .sort((a, b) => b.tasks - a.tasks)
    .slice(0, 10);

  return (
    <AnalyticsDashboard
      totalUsers={totalUsers}
      totalSites={totalSites}
      activeSites={activeSites}
      completedSites={completedSites}
      totalTasks={totalTasks}
      doneTasks={doneTasks}
      pendingTasks={pendingTasks}
      totalReports={totalReports}
      totalWorkers={totalWorkers}
      approvedWorkers={approvedWorkers}
      pendingWorkers={pendingWorkers}
      sitesByDay={sitesByDay}
      tasksByDay={tasksByDay}
      reportsByDay={reportsByDay}
      usersByDay={usersByDay}
      rolesDistribution={rolesDistribution}
      taskStatusDistribution={taskStatusDistribution}
      sitesByTasks={sitesByTasks}
    />
  );
}

