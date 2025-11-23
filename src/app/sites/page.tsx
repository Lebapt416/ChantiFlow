import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { CreateSiteCard } from './create-site-card';
import { SiteCard } from './site-card';
import { getUserPlan, getPlanLimits, canCreateSite, getUserAddOns } from '@/lib/plans';

export const metadata = {
  title: 'Chantiers | ChantiFlow',
};

export default async function SitesPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: sites } = await supabase
    .from('sites')
    .select('id, name, deadline, created_at, completed_at')
    .eq('created_by', user.id)
    .order('created_at', { ascending: false });

  // Séparer les chantiers actifs et terminés
  const activeSites = sites?.filter((site) => !site.completed_at) ?? [];

  // Récupérer les stats pour chaque chantier
  const siteIds = sites?.map((site) => site.id) ?? [];
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

  // Vérifier les limites du plan
  const plan = await getUserPlan(user);
  const addOns = getUserAddOns(user);
  const limits = getPlanLimits(plan, addOns);
  const { allowed: canCreate, reason: limitReason } = await canCreateSite(user.id);
  // Compter uniquement les chantiers actifs (non terminés)
  const currentCount = activeSites.length;

  return (
    <AppShell
      heading="Chantiers"
      subheading="Tous vos chantiers en un coup d'œil"
      userEmail={user.email}
    >
      <div className="overflow-x-auto pb-4">
        <div className="flex gap-4 min-w-max">
          {sites?.map((site) => {
            const stats = siteStats[site.id] || { tasks: 0, done: 0, workers: 0 };
            return (
              <SiteCard key={site.id} site={site} stats={stats} />
            );
          })}

          {/* Carte pour créer un nouveau chantier */}
          <CreateSiteCard
            canCreate={canCreate}
            limitReason={limitReason}
            currentCount={currentCount}
            maxSites={limits.maxSites}
          />
        </div>
      </div>
    </AppShell>
  );
}

