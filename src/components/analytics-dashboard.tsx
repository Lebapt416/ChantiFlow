'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter, usePathname } from 'next/navigation';
import { LayoutDashboard, User, Trash2, TestTube } from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

type AnalyticsDashboardProps = {
  totalUsers: number;
  totalSites: number;
  activeSites: number;
  completedSites: number;
  totalTasks: number;
  doneTasks: number;
  pendingTasks: number;
  totalReports: number;
  totalWorkers: number;
  approvedWorkers: number;
  pendingWorkers: number;
  sitesByDay: Array<{ date: string; sites: number }>;
  tasksByDay: Array<{ date: string; tasks: number }>;
  reportsByDay: Array<{ date: string; reports: number }>;
  usersByDay: Array<{ date: string; users: number }>;
  rolesDistribution: Record<string, number>;
  taskStatusDistribution: Record<string, number>;
  sitesByTasks: Array<{ name: string; tasks: number; done: number; progress: number; workers: number }>;
  completionRate: number;
  avgTasksPerSite: number;
  avgWorkersPerSite: number;
  reportsWithPhotos: number;
  reportsWithoutPhotos: number;
  totalHours: number;
  completedHours: number;
  topCreators: Array<{ name: string; sites: number }>;
  sitesByWeek: Array<{ week: string; sites: number }>;
  tasksByWeek: Array<{ week: string; tasks: number }>;
  tasksByRoleData: Array<{ name: string; value: number }>;
  sitesGrowth: number;
  tasksGrowth: number;
  mrr: number;
  plusUsers: number;
  proUsers: number;
  basicUsers: number;
  mrrByDay: Array<{ date: string; mrr: number; plus: number; pro: number }>;
  contactMessages: Array<{
    id: string;
    name: string;
    email: string;
    company: string | null;
    message: string;
    created_at: string;
  }>;
};

const COLORS = ['#10b981', '#f59e0b', '#3b82f6', '#ef4444', '#8b5cf6', '#ec4899'];

type ContactMessage = {
  id: string;
  name: string;
  email: string;
  company: string | null;
  message: string;
  created_at: string;
};

export function AnalyticsDashboard({
  totalUsers,
  totalSites,
  activeSites,
  completedSites,
  totalTasks,
  doneTasks,
  pendingTasks,
  totalReports,
  totalWorkers,
  approvedWorkers,
  pendingWorkers,
  sitesByDay,
  tasksByDay,
  reportsByDay,
  usersByDay,
  rolesDistribution,
  taskStatusDistribution,
  sitesByTasks,
  completionRate,
  avgTasksPerSite,
  avgWorkersPerSite,
  reportsWithPhotos,
  reportsWithoutPhotos,
  totalHours,
  completedHours,
  topCreators,
  sitesByWeek,
  tasksByWeek,
  tasksByRoleData,
  sitesGrowth,
  tasksGrowth,
  mrr,
  plusUsers,
  proUsers,
  basicUsers,
  mrrByDay,
  contactMessages: initialContactMessages,
}: AnalyticsDashboardProps) {
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [contactMessages, setContactMessages] = useState<ContactMessage[]>(initialContactMessages);
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());
  const router = useRouter();
  const pathname = usePathname();

  // Actualiser les données toutes les 30 secondes (uniquement sur la page analytics principale)
  useEffect(() => {
    // Ne pas recharger si on est sur la page des tests
    if (pathname?.includes('/system-test')) {
      return;
    }

    // DÉSACTIVÉ : Le reload automatique cause des problèmes de performance
    // L'utilisateur peut rafraîchir manuellement si nécessaire
    // const interval = setInterval(() => {
    //   setLastUpdate(new Date());
    //   // Ne recharger que si on est toujours sur la page analytics principale
    //   if (!window.location.pathname.includes('/system-test')) {
    //     window.location.reload();
    //   }
    // }, 30000);
    
    // Mise à jour de l'heure uniquement, sans reload
    const interval = setInterval(() => {
      setLastUpdate(new Date());
    }, 60000); // Mise à jour toutes les minutes sans reload

    return () => clearInterval(interval);
  }, [pathname]);

  // Fonction pour supprimer un message
  const handleDeleteMessage = async (messageId: string) => {
    if (!confirm('Êtes-vous sûr de vouloir supprimer ce message ?')) {
      return;
    }

    setDeletingIds((prev) => new Set(prev).add(messageId));

    try {
      const response = await fetch(`/api/contact/delete?id=${messageId}`, {
        method: 'DELETE',
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Erreur lors de la suppression');
      }

      // Retirer le message de la liste
      setContactMessages((prev) => prev.filter((msg) => msg.id !== messageId));
    } catch (error) {
      console.error('Erreur suppression:', error);
      alert('Erreur lors de la suppression du message. Veuillez réessayer.');
    } finally {
      setDeletingIds((prev) => {
        const newSet = new Set(prev);
        newSet.delete(messageId);
        return newSet;
      });
    }
  };

  // Formater les dates pour les graphiques
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return `${date.getDate()}/${date.getMonth() + 1}`;
  };


  // Données combinées pour le graphique d'activité
  const activityData = sitesByDay.map((site, index) => ({
    date: formatDate(site.date),
    Sites: site.sites,
    Tâches: tasksByDay[index]?.tasks ?? 0,
    Rapports: reportsByDay[index]?.reports ?? 0,
    Utilisateurs: usersByDay[index]?.users ?? 0,
  }));

  // Données pour le graphique des rôles
  const rolesData = Object.entries(rolesDistribution).map(([name, value]) => ({
    name,
    value,
  }));

  // Données pour le graphique des statuts de tâches
  const taskStatusData = Object.entries(taskStatusDistribution).map(([name, value]) => ({
    name,
    value,
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950 text-white">
      {/* Menu latéral */}
      <aside className="fixed inset-y-0 left-0 z-20 w-16 flex-col items-center border-r border-zinc-800 bg-black/80 px-0 py-8 shadow-xl backdrop-blur flex">
        <nav className="flex flex-1 flex-col items-center gap-2 w-full">
          <Link
            href="/analytics"
            className="group/item relative flex items-center justify-center w-14 h-14 rounded-xl transition-all duration-200 bg-white text-black shadow-lg shadow-white/20"
            title="Analytics"
          >
            <span className="absolute rounded-xl transition-all duration-200 inset-0 bg-white"></span>
            <span className="relative z-10">
              <LayoutDashboard size={26} strokeWidth={3.5} />
            </span>
          </Link>
          <Link
            href="/analytics/profile"
            className="group/item relative flex items-center justify-center w-14 h-14 rounded-xl transition-all duration-200 text-white hover:text-white"
            title="Profil"
          >
            <span className="absolute rounded-xl transition-all duration-200 top-0 bottom-0 left-2 right-0 bg-black/50 group-hover/item:bg-black/70 group-hover/item:left-3"></span>
            <span className="relative z-10">
              <User size={26} strokeWidth={3} className="group-hover/item:scale-110 transition-transform duration-200" />
            </span>
          </Link>
        </nav>
      </aside>
      <div className="ml-16">
        <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-2">Analytics ChantiFlow</h1>
              <p className="text-zinc-400">
                Données en temps réel • Dernière mise à jour :{' '}
                {lastUpdate.toLocaleTimeString('fr-FR')}
              </p>
            </div>
            <div className="text-right">
              <div className="inline-flex items-center gap-2 rounded-full bg-emerald-500/20 px-4 py-2">
                <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></div>
                <span className="text-sm font-semibold text-emerald-400">En direct</span>
              </div>
            </div>
          </div>
        </div>

        {/* Lien vers les tests système */}
        <div className="mb-8">
          <Link
            href="/analytics/system-test"
            className="block rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur hover:bg-zinc-900/70 transition-colors group"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-xl bg-blue-500/10 border border-blue-500/30 group-hover:bg-blue-500/20 transition-colors">
                  <TestTube className="h-6 w-6 text-blue-400" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white group-hover:text-blue-400 transition-colors">
                    Tests Système
                  </h2>
                  <p className="mt-1 text-sm text-zinc-400">
                    Vérifiez l'état de fonctionnement de tous les composants avec historique
                  </p>
                </div>
              </div>
              <div className="text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity">
                →
              </div>
            </div>
          </Link>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <p className="text-sm text-zinc-400 mb-2">Utilisateurs</p>
            <p className="text-3xl font-bold text-white">{totalUsers}</p>
          </div>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <p className="text-sm text-zinc-400 mb-2">Chantiers</p>
            <p className="text-3xl font-bold text-white">
              {activeSites} <span className="text-lg text-zinc-500">/ {totalSites}</span>
            </p>
            <p className="text-xs text-zinc-500 mt-1">
              {completedSites} terminés
            </p>
            {sitesGrowth !== 0 && (
              <p className={`text-xs mt-1 ${sitesGrowth > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {sitesGrowth > 0 ? '↑' : '↓'} {Math.abs(sitesGrowth)}% vs semaine dernière
              </p>
            )}
          </div>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <p className="text-sm text-zinc-400 mb-2">Tâches</p>
            <p className="text-3xl font-bold text-white">
              {doneTasks} <span className="text-lg text-zinc-500">/ {totalTasks}</span>
            </p>
            <p className="text-xs text-zinc-500 mt-1">
              {pendingTasks} en attente • {completionRate}% complété
            </p>
            {tasksGrowth !== 0 && (
              <p className={`text-xs mt-1 ${tasksGrowth > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {tasksGrowth > 0 ? '↑' : '↓'} {Math.abs(tasksGrowth)}% vs semaine dernière
              </p>
            )}
          </div>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <p className="text-sm text-zinc-400 mb-2">Rapports</p>
            <p className="text-3xl font-bold text-white">{totalReports}</p>
            <p className="text-xs text-zinc-500 mt-1">
              {reportsWithPhotos} avec photos
            </p>
          </div>
        </div>

        {/* MRR Card - Prominent */}
        <div className="rounded-2xl border-2 border-emerald-500/50 bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 p-6 backdrop-blur mb-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <p className="text-sm text-emerald-400 mb-2 font-semibold">MRR (Monthly Recurring Revenue)</p>
              <p className="text-5xl font-bold text-white">{mrr}€</p>
              <p className="text-sm text-zinc-400 mt-2">
                {plusUsers} Plus (29€) + {proUsers} Pro (79€) = {mrr}€/mois
              </p>
            </div>
            <div className="text-right">
              <div className="space-y-2">
                <div className="rounded-lg bg-zinc-800/50 px-4 py-2">
                  <p className="text-xs text-zinc-400">Basic</p>
                  <p className="text-xl font-bold text-white">{basicUsers}</p>
                </div>
                <div className="rounded-lg bg-blue-500/20 px-4 py-2 border border-blue-500/30">
                  <p className="text-xs text-blue-400">Plus</p>
                  <p className="text-xl font-bold text-white">{plusUsers}</p>
                </div>
                <div className="rounded-lg bg-purple-500/20 px-4 py-2 border border-purple-500/30">
                  <p className="text-xs text-purple-400">Pro</p>
                  <p className="text-xl font-bold text-white">{proUsers}</p>
                </div>
              </div>
            </div>
          </div>
          {/* Graphique MRR */}
          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-4 text-white">Évolution du MRR (30 derniers jours)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={mrrByDay.map((day, index) => ({
                date: formatDate(day.date),
                MRR: day.mrr,
                Plus: day.plus * 29,
                Pro: day.pro * 79,
              }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="date" 
                  stroke="#9ca3af"
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                />
                <YAxis 
                  stroke="#9ca3af"
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                  label={{ value: '€', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                  formatter={(value: number) => `${value}€`}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="MRR"
                  stroke="#10b981"
                  strokeWidth={3}
                  dot={{ fill: '#10b981', r: 5 }}
                  activeDot={{ r: 7 }}
                />
                <Line
                  type="monotone"
                  dataKey="Plus"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={{ fill: '#3b82f6', r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="Pro"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={{ fill: '#8b5cf6', r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Additional Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <p className="text-sm text-zinc-400 mb-2">Taux de complétion</p>
            <p className="text-3xl font-bold text-white">{completionRate}%</p>
            <div className="mt-2 h-2 bg-zinc-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-emerald-500 transition-all duration-500"
                style={{ width: `${completionRate}%` }}
              ></div>
            </div>
          </div>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <p className="text-sm text-zinc-400 mb-2">Moyenne tâches/chantier</p>
            <p className="text-3xl font-bold text-white">{avgTasksPerSite}</p>
            <p className="text-xs text-zinc-500 mt-1">
              {avgWorkersPerSite} workers/chantier
            </p>
          </div>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <p className="text-sm text-zinc-400 mb-2">Heures de travail</p>
            <p className="text-3xl font-bold text-white">
              {completedHours} <span className="text-lg text-zinc-500">/ {totalHours}h</span>
            </p>
            <p className="text-xs text-zinc-500 mt-1">
              {totalHours > 0 ? Math.round((completedHours / totalHours) * 100) : 0}% complété
            </p>
          </div>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <p className="text-sm text-zinc-400 mb-2">Workers</p>
            <p className="text-3xl font-bold text-white">{totalWorkers}</p>
            <p className="text-xs text-zinc-500 mt-1">
              {approvedWorkers} approuvés • {pendingWorkers} en attente
            </p>
          </div>
        </div>

        {/* Workers Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <p className="text-sm text-zinc-400 mb-2">Workers</p>
            <p className="text-3xl font-bold text-white">{totalWorkers}</p>
            <div className="mt-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-zinc-400">Approuvés</span>
                <span className="text-lg font-semibold text-emerald-400">{approvedWorkers}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-zinc-400">En attente</span>
                <span className="text-lg font-semibold text-amber-400">{pendingWorkers}</span>
              </div>
            </div>
          </div>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <p className="text-sm text-zinc-400 mb-4">Répartition des rôles</p>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={rolesData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${((percent ?? 0) * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {rolesData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Activity Chart */}
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur mb-8">
          <h2 className="text-xl font-semibold mb-4">Activité des 30 derniers jours</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={activityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="Sites"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ fill: '#3b82f6', r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="Tâches"
                stroke="#10b981"
                strokeWidth={2}
                dot={{ fill: '#10b981', r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="Rapports"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={{ fill: '#f59e0b', r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="Utilisateurs"
                stroke="#ef4444"
                strokeWidth={2}
                dot={{ fill: '#ef4444', r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Task Status Chart */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <h2 className="text-xl font-semibold mb-4">Statut des tâches</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={taskStatusData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value, percent }) =>
                    `${name}: ${value} (${((percent ?? 0) * 100).toFixed(0)}%)`
                  }
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {taskStatusData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={index === 0 ? '#10b981' : '#f59e0b'}
                    />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <h2 className="text-xl font-semibold mb-4">Top 10 chantiers par tâches</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={sitesByTasks}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="name"
                  stroke="#9ca3af"
                  angle={-45}
                  textAnchor="end"
                  height={100}
                  interval={0}
                />
                <YAxis stroke="#9ca3af" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="tasks" fill="#3b82f6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Weekly Activity */}
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur mb-8">
          <h2 className="text-xl font-semibold mb-4">Activité hebdomadaire (7 dernières semaines)</h2>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={sitesByWeek.map((site, index) => ({
              week: `S${index + 1}`,
              Sites: site.sites,
              Tâches: tasksByWeek[index]?.tasks ?? 0,
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="week" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
              />
              <Legend />
              <Bar dataKey="Sites" fill="#3b82f6" radius={[8, 8, 0, 0]} />
              <Bar dataKey="Tâches" fill="#10b981" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Tasks by Role */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <h2 className="text-xl font-semibold mb-4">Tâches par rôle requis</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={tasksByRoleData.slice(0, 10)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="name"
                  stroke="#9ca3af"
                  angle={-45}
                  textAnchor="end"
                  height={100}
                />
                <YAxis stroke="#9ca3af" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="value" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <h2 className="text-xl font-semibold mb-4">Top 5 créateurs de chantiers</h2>
            <div className="space-y-4">
              {topCreators.map((creator, index) => (
                <div key={index} className="flex items-center justify-between p-4 rounded-lg bg-zinc-800/50">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400 font-bold">
                      {index + 1}
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-white">{creator.name}</p>
                      <p className="text-xs text-zinc-400">{creator.sites} chantier{creator.sites > 1 ? 's' : ''}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-bold text-emerald-400">{creator.sites}</p>
                  </div>
                </div>
              ))}
              {topCreators.length === 0 && (
                <p className="text-zinc-400 text-center py-8">Aucune donnée disponible</p>
              )}
            </div>
          </div>
        </div>

        {/* Detailed Sites Table */}
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur mb-8">
          <h2 className="text-xl font-semibold mb-4">Détails des chantiers</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-zinc-700">
                  <th className="text-left py-3 px-4 text-sm font-semibold text-zinc-400">Chantier</th>
                  <th className="text-right py-3 px-4 text-sm font-semibold text-zinc-400">Tâches</th>
                  <th className="text-right py-3 px-4 text-sm font-semibold text-zinc-400">Terminées</th>
                  <th className="text-right py-3 px-4 text-sm font-semibold text-zinc-400">Workers</th>
                  <th className="text-right py-3 px-4 text-sm font-semibold text-zinc-400">Progression</th>
                </tr>
              </thead>
              <tbody>
                {sitesByTasks.map((site, index) => (
                  <tr key={index} className="border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors">
                    <td className="py-3 px-4 text-white font-medium">{site.name}</td>
                    <td className="py-3 px-4 text-right text-white">{site.tasks}</td>
                    <td className="py-3 px-4 text-right text-emerald-400">{site.done}</td>
                    <td className="py-3 px-4 text-right text-white">{site.workers}</td>
                    <td className="py-3 px-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-24 h-2 bg-zinc-800 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-emerald-500 transition-all duration-500"
                            style={{ width: `${site.progress}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-zinc-400 w-12 text-right">{site.progress}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
                {sitesByTasks.length === 0 && (
                  <tr>
                    <td colSpan={5} className="py-8 text-center text-zinc-400">
                      Aucun chantier disponible
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Section Messages de Contact */}
      <div className="mb-8">
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-white">Messages de Contact</h2>
              <p className="mt-1 text-sm text-zinc-400">
                {contactMessages.length} message{contactMessages.length > 1 ? 's' : ''} reçu{contactMessages.length > 1 ? 's' : ''}
              </p>
            </div>
          </div>

          <div className="space-y-4 max-h-[600px] overflow-y-auto">
            {contactMessages.length > 0 ? (
              contactMessages.map((msg) => (
                <div
                  key={msg.id}
                  className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-5 hover:bg-zinc-900/50 transition-colors"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-lg font-semibold text-white">{msg.name}</h3>
                        {msg.company && (
                          <span className="text-xs px-2 py-1 rounded-full bg-zinc-800 text-zinc-300">
                            {msg.company}
                          </span>
                        )}
                      </div>
                      <a
                        href={`mailto:${msg.email}`}
                        className="text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
                      >
                        {msg.email}
                      </a>
                    </div>
                    <span className="text-xs text-zinc-500">
                      {new Date(msg.created_at).toLocaleDateString('fr-FR', {
                        day: '2-digit',
                        month: 'short',
                        year: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </span>
                  </div>
                  <div className="mt-3 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800">
                    <p className="text-sm text-zinc-300 whitespace-pre-wrap leading-relaxed">
                      {msg.message}
                    </p>
                  </div>
                  <div className="mt-3 flex items-center gap-2">
                    <a
                      href={`mailto:${msg.email}?subject=Re: Votre message de contact&cc=chantiflowct@gmail.com`}
                      className="text-xs px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition-colors"
                      title="Répondre (utilisez chantiflowct@gmail.com comme expéditeur)"
                    >
                      Répondre
                    </a>
                    <button
                      onClick={() => handleDeleteMessage(msg.id)}
                      disabled={deletingIds.has(msg.id)}
                      className="text-xs px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
                      title="Supprimer ce message"
                    >
                      {deletingIds.has(msg.id) ? (
                        <>
                          <div className="h-3 w-3 animate-spin rounded-full border-2 border-white border-t-transparent" />
                          Suppression...
                        </>
                      ) : (
                        <>
                          <Trash2 className="h-3 w-3" />
                          Supprimer
                        </>
                      )}
                    </button>
                    <span className="text-xs text-zinc-400 self-center ml-auto">
                      Depuis: chantiflowct@gmail.com
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-12 text-zinc-400">
                <p>Aucun message de contact pour le moment</p>
              </div>
            )}
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}

