'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { LayoutDashboard, User } from 'lucide-react';
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
  sitesByTasks: Array<{ name: string; tasks: number }>;
};

const COLORS = ['#10b981', '#f59e0b', '#3b82f6', '#ef4444', '#8b5cf6', '#ec4899'];

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
}: AnalyticsDashboardProps) {
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Actualiser les données toutes les 30 secondes
  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdate(new Date());
      window.location.reload();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

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
          </div>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <p className="text-sm text-zinc-400 mb-2">Tâches</p>
            <p className="text-3xl font-bold text-white">
              {doneTasks} <span className="text-lg text-zinc-500">/ {totalTasks}</span>
            </p>
            <p className="text-xs text-zinc-500 mt-1">
              {pendingTasks} en attente
            </p>
          </div>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
            <p className="text-sm text-zinc-400 mb-2">Rapports</p>
            <p className="text-3xl font-bold text-white">{totalReports}</p>
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
      </div>
      </div>
    </div>
  );
}

