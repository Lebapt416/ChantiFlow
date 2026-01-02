import { createSupabaseServerClient } from '@/lib/supabase/server';
import { SiteCard } from '@/app/sites/site-card';
import { CreateSiteCard } from '@/app/sites/create-site-card';
import { getUserPlan, getPlanLimits, canCreateSite, getUserAddOns } from '@/lib/plans';
import { CheckCircle2 } from 'lucide-react';

type DashboardContentProps = {
  userId: string;
  userEmail: string;
  userMetadata?: Record<string, unknown> | null;
};

export async function DashboardContent({ userId, userEmail, userMetadata }: DashboardContentProps) {
  const supabase = await createSupabaseServerClient();

  // Optimisation : Requêtes en parallèle avec caching
  const [{ data: sites }] = await Promise.all([
    supabase
      .from('sites')
      .select('id, name, deadline, created_at, completed_at')
      .eq('created_by', userId)
      .order('created_at', { ascending: false })
      .maybeSingle(), // Utiliser maybeSingle pour éviter les erreurs si vide
  ]);

  // Récupérer les stats pour chaque chantier
  const sitesArray = Array.isArray(sites) ? sites : (sites ? [sites] : []);
  const siteIds = sitesArray.map((site) => site.id);
  const siteStats: Record<string, { tasks: number; done: number; workers: number }> = {};

  if (siteIds.length > 0) {
    const [{ data: tasks }, { data: workers }] = await Promise.all([
      supabase
        .from('tasks')
        .select('site_id, status')
        .in('site_id', siteIds),
      supabase
        .from('workers')
        .select('site_id')
        .in('site_id', siteIds),
    ]);

    tasks?.forEach((task) => {
      if (!siteStats[task.site_id]) {
        siteStats[task.site_id] = { tasks: 0, done: 0, workers: 0 };
      }
      siteStats[task.site_id].tasks++;
      if (task.status === 'done') {
        siteStats[task.site_id].done++;
      }
    });

    workers?.forEach((worker) => {
      if (!siteStats[worker.site_id]) {
        siteStats[worker.site_id] = { tasks: 0, done: 0, workers: 0 };
      }
      siteStats[worker.site_id].workers++;
    });
  }

  // Séparer les chantiers actifs et terminés
  const activeSites = sitesArray.filter((site) => !site.completed_at);
  const completedSites = sitesArray.filter((site) => site.completed_at);

  // Vérifier les limites du plan (en parallèle)
  const user = { id: userId, user_metadata: userMetadata ?? undefined };
  const [plan, addOns, canCreateData] = await Promise.all([
    getUserPlan(user),
    getUserAddOns(user),
    canCreateSite(userId),
  ]);
  const limits = getPlanLimits(plan, addOns);
  const { allowed: canCreate, reason: limitReason } = canCreateData;
  // Compter uniquement les chantiers actifs (non terminés)
  const currentCount = activeSites.length;

  return (
    <>
      {/* Chantiers en cours */}
      {activeSites.length > 0 && (
        <div className="mb-8">
          <div className="mb-4 flex items-center gap-2">
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">
              Chantiers en cours
            </h2>
            <span className="rounded-full bg-emerald-100 px-2 py-1 text-xs font-semibold text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
              {activeSites.length}
            </span>
          </div>
          <div className="overflow-x-auto pb-4">
            <div className="flex gap-4 min-w-max">
              {activeSites.map((site) => {
                const stats = siteStats[site.id] || { tasks: 0, done: 0, workers: 0 };
                return (
                  <SiteCard key={site.id} site={site} stats={stats} />
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Chantiers terminés */}
      {completedSites.length > 0 && (
        <div className="mb-8">
          <div className="mb-4 flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">
              Chantiers terminés
            </h2>
            <span className="rounded-full bg-zinc-100 px-2 py-1 text-xs font-semibold text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
              {completedSites.length}
            </span>
          </div>
          <div className="overflow-x-auto pb-4">
            <div className="flex gap-4 min-w-max">
              {completedSites.map((site) => {
                const stats = siteStats[site.id] || { tasks: 0, done: 0, workers: 0 };
                return (
                  <SiteCard key={site.id} site={site} stats={stats} />
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Carte pour créer un nouveau chantier */}
      <div className="overflow-x-auto pb-4">
        <div className="flex gap-4 min-w-max">
          <CreateSiteCard
            canCreate={canCreate}
            limitReason={limitReason}
            currentCount={currentCount}
            maxSites={limits.maxSites}
          />
        </div>
      </div>

      {sitesArray.length === 0 && (
        <div className="rounded-2xl border border-zinc-200 bg-white p-8 text-center dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Aucun chantier pour le moment. Créez votre premier chantier pour commencer.
          </p>
        </div>
      )}
    </>
  );
}

