import { createSupabaseServerClient } from '@/lib/supabase/server';
import { SiteCard } from '@/app/sites/site-card';
import { CreateSiteCard } from '@/app/sites/create-site-card';
import { getUserPlan, getPlanLimits, canCreateSite, getUserAddOns } from '@/lib/plans';

type DashboardContentProps = {
  userId: string;
  userEmail: string;
  userMetadata?: Record<string, unknown> | null;
};

export async function DashboardContent({ userId, userMetadata }: DashboardContentProps) {
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

  // Total tasks in progress across all active sites
  const totalTasksInProgress = activeSites.reduce((sum, site) => {
    const stats = siteStats[site.id];
    return sum + (stats ? stats.tasks - stats.done : 0);
  }, 0);

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
      {/* Stats bar */}
      <div className="grid grid-cols-3 border-b border-rule-soft mb-10 divide-x divide-rule-soft">
        <div className="py-6 pr-8">
          <div className="font-serif text-[48px] leading-none tracking-tight text-ink" style={{fontVariationSettings: '"opsz" 144'}}>
            {activeSites.length}
          </div>
          <div className="mt-2 font-mono text-[10px] uppercase tracking-widest text-ink-3">Chantiers actifs</div>
        </div>
        <div className="py-6 px-8">
          <div className="font-serif text-[48px] leading-none tracking-tight text-ink" style={{fontVariationSettings: '"opsz" 144'}}>
            {totalTasksInProgress}
          </div>
          <div className="mt-2 font-mono text-[10px] uppercase tracking-widest text-ink-3">Tâches en cours</div>
        </div>
        <div className="py-6 pl-8">
          <div className="font-serif text-[48px] leading-none tracking-tight text-ink" style={{fontVariationSettings: '"opsz" 144'}}>
            {completedSites.length}
          </div>
          <div className="mt-2 font-mono text-[10px] uppercase tracking-widest text-ink-3">Terminés</div>
        </div>
      </div>

      {/* Plan limit banner */}
      {!canCreate && limitReason && (
        <div className="border border-warn bg-paper-2 px-4 py-3 font-mono text-[11px] tracking-widest text-warn uppercase mb-6">
          ⚠ Limite de plan atteinte · {limitReason}
        </div>
      )}

      {/* Chantiers en cours */}
      {activeSites.length > 0 && (
        <div className="mb-8">
          <div className="mb-4 flex items-center gap-3">
            <p className="font-mono text-[11px] uppercase tracking-widest text-ink-3">Chantiers en cours</p>
            <span className="rounded-sm border border-rule-soft font-mono text-[10px] uppercase tracking-widest px-2 py-0.5 text-ink-3">
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
          <div className="mb-4 flex items-center gap-3">
            <p className="font-mono text-[11px] uppercase tracking-widest text-ink-3">Chantiers terminés</p>
            <span className="rounded-sm border border-rule-soft font-mono text-[10px] uppercase tracking-widest px-2 py-0.5 text-ink-3">
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
        <div className="border border-rule-soft bg-paper p-8 text-center">
          <p className="text-sm text-ink-3">
            Aucun chantier pour le moment. Créez votre premier chantier pour commencer.
          </p>
        </div>
      )}
    </>
  );
}
