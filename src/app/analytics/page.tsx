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
    error: authError,
  } = await supabase.auth.getUser();

  // Vérifier que l'utilisateur est connecté
  if (authError || !user) {
    redirect('/login?redirect=/analytics');
  }

  // Vérifier que l'utilisateur est le compte autorisé (par ID ou email)
  const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
  if (user.id !== authorizedUserId && user.email !== 'bcb83@icloud.com') {
    redirect('/login?error=unauthorized');
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
      adminClient.from('reports').select('id, task_id, worker_id, created_at, photo_url, description'),
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

  // Calculer le MRR (Monthly Recurring Revenue)
  const plusUsers = allUsers.filter((user) => user.user_metadata?.plan === 'plus').length;
  const proUsers = allUsers.filter((user) => user.user_metadata?.plan === 'pro').length;
  const basicUsers = allUsers.filter((user) => {
    const plan = user.user_metadata?.plan;
    return !plan || plan === 'basic';
  }).length;
  const mrr = (plusUsers * 29) + (proUsers * 79);

  // Calculer l'évolution du MRR sur les 30 derniers jours
  const mrrByDay = last30Days.map((day) => {
    // Pour chaque jour, compter les utilisateurs qui existaient déjà et avaient un plan payant
    const usersBeforeDay = allUsers.filter((user) => {
      const userDate = new Date(user.created_at).toISOString().split('T')[0];
      return userDate <= day;
    });

    const plusCount = usersBeforeDay.filter((user) => user.user_metadata?.plan === 'plus').length;
    const proCount = usersBeforeDay.filter((user) => user.user_metadata?.plan === 'pro').length;
    const dayMrr = (plusCount * 29) + (proCount * 79);

    return { date: day, mrr: dayMrr, plus: plusCount, pro: proCount };
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
      const siteTasks = allTasks.filter((task) => task.site_id === site.id);
      const taskCount = siteTasks.length;
      const doneCount = siteTasks.filter((t) => t.status === 'done').length;
      const progress = taskCount > 0 ? Math.round((doneCount / taskCount) * 100) : 0;
      return {
        name: site.name,
        tasks: taskCount,
        done: doneCount,
        progress,
        workers: allWorkers.filter((w) => w.site_id === site.id).length,
      };
    })
    .sort((a, b) => b.tasks - a.tasks)
    .slice(0, 10);

  // Statistiques supplémentaires
  const completionRate = totalTasks > 0 ? Math.round((doneTasks / totalTasks) * 100) : 0;
  const avgTasksPerSite = totalSites > 0 ? Math.round((totalTasks / totalSites) * 100) / 100 : 0;
  const avgWorkersPerSite = totalSites > 0 ? Math.round((totalWorkers / totalSites) * 100) / 100 : 0;
  const reportsWithPhotos = allReports.filter((r) => r.photo_url).length;
  const reportsWithoutPhotos = totalReports - reportsWithPhotos;
  
  // Total heures de travail estimées
  const totalHours = allTasks.reduce((sum, task) => sum + (task.duration_hours || 0), 0);
  const completedHours = allTasks
    .filter((t) => t.status === 'done')
    .reduce((sum, task) => sum + (task.duration_hours || 0), 0);

  // Répartition par créateur de chantier
  const sitesByCreator = allSites.reduce((acc, site) => {
    const creatorId = site.created_by || 'unknown';
    acc[creatorId] = (acc[creatorId] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  // Top créateurs de chantiers
  const topCreators = Object.entries(sitesByCreator)
    .map(([creatorId, count]) => {
      const creator = allUsers.find((u) => u.id === creatorId);
      return {
        name: creator?.email || creatorId.substring(0, 8) + '...',
        sites: count as number,
      };
    })
    .sort((a, b) => b.sites - a.sites)
    .slice(0, 5);

  // Statistiques par semaine (7 dernières semaines)
  const last7Weeks = Array.from({ length: 7 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (6 - i) * 7);
    const weekStart = new Date(date);
    weekStart.setDate(date.getDate() - date.getDay());
    return weekStart.toISOString().split('T')[0];
  });

  const sitesByWeek = last7Weeks.map((weekStart) => {
    const weekEnd = new Date(weekStart);
    weekEnd.setDate(weekEnd.getDate() + 7);
    const count = allSites.filter((site) => {
      const siteDate = new Date(site.created_at);
      return siteDate >= new Date(weekStart) && siteDate < weekEnd;
    }).length;
    return { week: weekStart, sites: count };
  });

  const tasksByWeek = last7Weeks.map((weekStart) => {
    const weekEnd = new Date(weekStart);
    weekEnd.setDate(weekEnd.getDate() + 7);
    const count = allTasks.filter((task) => {
      const taskDate = new Date(task.created_at);
      return taskDate >= new Date(weekStart) && taskDate < weekEnd;
    }).length;
    return { week: weekStart, tasks: count };
  });

  // Répartition des tâches par rôle requis
  const tasksByRole = allTasks.reduce((acc, task) => {
    const role = task.required_role || 'Non spécifié';
    acc[role] = (acc[role] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const tasksByRoleData = Object.entries(tasksByRole)
    .map(([name, value]) => ({ name, value: value as number }))
    .sort((a, b) => b.value - a.value);

  // Taux de croissance (comparaison avec la semaine précédente)
  const thisWeekSites = sitesByWeek[sitesByWeek.length - 1]?.sites || 0;
  const lastWeekSites = sitesByWeek[sitesByWeek.length - 2]?.sites || 0;
  const sitesGrowth = lastWeekSites > 0 
    ? Math.round(((thisWeekSites - lastWeekSites) / lastWeekSites) * 100) 
    : 0;

  const thisWeekTasks = tasksByWeek[tasksByWeek.length - 1]?.tasks || 0;
  const lastWeekTasks = tasksByWeek[tasksByWeek.length - 2]?.tasks || 0;
  const tasksGrowth = lastWeekTasks > 0 
    ? Math.round(((thisWeekTasks - lastWeekTasks) / lastWeekTasks) * 100) 
    : 0;

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
      completionRate={completionRate}
      avgTasksPerSite={avgTasksPerSite}
      avgWorkersPerSite={avgWorkersPerSite}
      reportsWithPhotos={reportsWithPhotos}
      reportsWithoutPhotos={reportsWithoutPhotos}
      totalHours={totalHours}
      completedHours={completedHours}
      topCreators={topCreators}
      sitesByWeek={sitesByWeek}
      tasksByWeek={tasksByWeek}
      tasksByRoleData={tasksByRoleData}
      sitesGrowth={sitesGrowth}
      tasksGrowth={tasksGrowth}
      mrr={mrr}
      plusUsers={plusUsers}
      proUsers={proUsers}
      basicUsers={basicUsers}
      mrrByDay={mrrByDay}
    />
  );
}

