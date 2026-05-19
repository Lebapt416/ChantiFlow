import Link from 'next/link';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { getUserPlan, getPlanLimits, canCreateSite, getUserAddOns } from '@/lib/plans';
import { DashboardNewSiteForm } from './dashboard-form';

type DashboardContentProps = {
  userId: string;
  userEmail: string;
  userMetadata?: Record<string, unknown> | null;
};

function fmt2(n: number) {
  return n < 10 ? `0${n}` : `${n}`;
}

function daysUntil(deadline: string): number {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const d = new Date(deadline);
  d.setHours(0, 0, 0, 0);
  return Math.round((d.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
}

export async function DashboardContent({ userId, userEmail, userMetadata }: DashboardContentProps) {
  const supabase = await createSupabaseServerClient();

  // Requête sans .maybeSingle() pour récupérer TOUS les chantiers
  const [{ data: sitesRaw }] = await Promise.all([
    supabase
      .from('sites')
      .select('id, name, deadline, address, postal_code, created_at, completed_at')
      .eq('created_by', userId)
      .order('created_at', { ascending: false }),
  ]);

  const sitesArray = sitesRaw ?? [];
  const siteIds = sitesArray.map((s) => s.id);
  const siteStats: Record<string, { tasks: number; done: number; workers: number }> = {};

  if (siteIds.length > 0) {
    const [{ data: tasks }, { data: workers }] = await Promise.all([
      supabase.from('tasks').select('site_id, status').in('site_id', siteIds),
      supabase.from('workers').select('site_id').in('site_id', siteIds),
    ]);

    tasks?.forEach((t) => {
      if (!siteStats[t.site_id]) siteStats[t.site_id] = { tasks: 0, done: 0, workers: 0 };
      siteStats[t.site_id].tasks++;
      if (t.status === 'done') siteStats[t.site_id].done++;
    });

    workers?.forEach((w) => {
      if (!siteStats[w.site_id]) siteStats[w.site_id] = { tasks: 0, done: 0, workers: 0 };
      siteStats[w.site_id].workers++;
    });
  }

  const activeSites = sitesArray.filter((s) => !s.completed_at);
  const completedSites = sitesArray.filter((s) => s.completed_at);

  // Stats globales
  const totalTasksAll = Object.values(siteStats).reduce((sum, s) => sum + s.tasks, 0);
  const totalDoneAll = Object.values(siteStats).reduce((sum, s) => sum + s.done, 0);
  const totalInProgress = activeSites.reduce((sum, s) => {
    const st = siteStats[s.id];
    return sum + (st ? st.tasks - st.done : 0);
  }, 0);
  const globalProgress = totalTasksAll > 0 ? Math.round((totalDoneAll / totalTasksAll) * 100) : 0;

  // Plan / limites
  const user = { id: userId, user_metadata: userMetadata ?? undefined };
  const [plan, addOns, canCreateData] = await Promise.all([
    getUserPlan(user),
    getUserAddOns(user),
    canCreateSite(userId),
  ]);
  const limits = getPlanLimits(plan, addOns);
  const { allowed: canCreate, reason: limitReason } = canCreateData;
  const currentCount = activeSites.length;

  // Deadlines à venir (chantiers avec deadline, triés ASC)
  const upcomingDeadlines = activeSites
    .filter((s) => s.deadline)
    .sort((a, b) => new Date(a.deadline!).getTime() - new Date(b.deadline!).getTime())
    .slice(0, 5);

  return (
    <div className="bg-paper min-h-screen">

      {/* ── TOPBAR ─────────────────────────────────────────────── */}
      <header className="flex items-start justify-between px-8 py-9 border-b border-rule lg:px-12">
        <div>
          <div className="font-mono text-[11px] uppercase tracking-widest text-ink-3 mb-3">
            ChantiFlow · {userEmail}
          </div>
          <h1
            className="font-serif text-[40px] leading-none tracking-tight text-ink lg:text-[52px]"
            style={{ fontVariationSettings: '"opsz" 144, "SOFT" 30' }}
          >
            Dashboard{' '}
            <em
              className="not-italic"
              style={{
                fontStyle: 'italic',
                color: 'var(--color-orange)',
                fontWeight: 300,
                fontVariationSettings: '"opsz" 144, "SOFT" 100',
              }}
            >
              général
            </em>
          </h1>
          <p className="text-[15px] text-ink-2 mt-2 max-w-[540px]">
            Vue d'ensemble de tous vos chantiers — chiffres clés, occupation et prochaines deadlines.
          </p>
        </div>
        <div className="hidden lg:flex items-center gap-3 mt-2">
          <span className="inline-flex items-center gap-2 px-3 py-2 border border-rule font-mono text-[11px] uppercase tracking-widest text-ink">
            <span
              className="w-1.5 h-1.5 rounded-full bg-green animate-pulse"
              style={{ animationDuration: '2s' }}
            />
            IA connectée
          </span>
          <Link
            href="/account"
            className="inline-flex items-center px-3 py-2 border border-rule font-mono text-[11px] uppercase tracking-widest text-ink hover:bg-ink hover:text-paper transition-colors"
          >
            Mon compte →
          </Link>
        </div>
      </header>

      {/* ── STATS STRIP ──────────────────────────────────────── */}
      <section className="grid grid-cols-3 border-b border-rule divide-x divide-rule-soft">

        {/* 01 · Chantiers */}
        <div className="relative p-8 lg:p-12">
          <span className="absolute top-4 right-5 font-mono text-[10px] text-ink-3 tracking-widest">01</span>
          <div className="font-mono text-[11px] uppercase tracking-widest text-ink-2 mb-4">Chantiers</div>
          <div
            className="font-serif leading-[0.95] tracking-tight text-ink text-[52px] lg:text-[72px]"
            style={{ fontVariationSettings: '"opsz" 144, "SOFT" 30' }}
          >
            {fmt2(activeSites.length)}
            <span className="text-[0.5em] opacity-50 align-super"> / {fmt2(sitesArray.length)}</span>
          </div>
          <div className="font-mono text-[13px] text-ink-2 mt-3">
            {completedSites.length} terminé{completedSites.length !== 1 ? 's' : ''} · {activeSites.length} actif{activeSites.length !== 1 ? 's' : ''}
          </div>
        </div>

        {/* 02 · Tâches */}
        <div className="relative p-8 lg:p-12">
          <span className="absolute top-4 right-5 font-mono text-[10px] text-ink-3 tracking-widest">02</span>
          <div className="font-mono text-[11px] uppercase tracking-widest text-ink-2 mb-4">Tâches</div>
          <div
            className="font-serif leading-[0.95] tracking-tight text-ink text-[52px] lg:text-[72px]"
            style={{ fontVariationSettings: '"opsz" 144, "SOFT" 30' }}
          >
            {fmt2(totalInProgress)}
          </div>
          <div className="font-mono text-[13px] text-ink-2 mt-3">
            {totalInProgress} en attente · {totalDoneAll} terminées
          </div>
        </div>

        {/* 03 · Progression */}
        <div className="relative p-8 lg:p-12">
          <span className="absolute top-4 right-5 font-mono text-[10px] text-ink-3 tracking-widest">03</span>
          <div className="font-mono text-[11px] uppercase tracking-widest text-ink-2 mb-4">Progression</div>
          <div
            className="font-serif leading-[0.95] tracking-tight text-[52px] lg:text-[72px]"
            style={{ fontVariationSettings: '"opsz" 144, "SOFT" 30' }}
          >
            <em
              className="not-italic"
              style={{
                fontStyle: 'italic',
                color: 'var(--color-orange)',
                fontWeight: 300,
                fontVariationSettings: '"opsz" 144, "SOFT" 100',
              }}
            >
              {globalProgress}
            </em>
            <span className="text-[0.45em] opacity-50 align-super text-ink">%</span>
          </div>
          {/* Barre orange */}
          <div className="mt-4 h-1.5 bg-paper-2 relative overflow-hidden">
            <div
              className="absolute inset-y-0 left-0 bg-orange transition-all duration-500"
              style={{ width: `${globalProgress}%` }}
            />
          </div>
        </div>

      </section>

      {/* ── CONTENT GRID ─────────────────────────────────────── */}
      <section className="grid grid-cols-1 lg:grid-cols-2 divide-x divide-rule-soft">

        {/* ── COL GAUCHE : formulaire ── */}
        <div className="p-8 lg:p-12">
          {/* En-tête section */}
          <div className="flex items-end justify-between mb-8 pb-5 border-b border-rule-soft">
            <div>
              <div className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-2">
                Section 01 · Création
              </div>
              <h2
                className="font-serif text-[28px] leading-none tracking-tight text-ink lg:text-[32px]"
                style={{ fontVariationSettings: '"opsz" 60, "SOFT" 30' }}
              >
                Ajouter un{' '}
                <em
                  className="not-italic"
                  style={{
                    fontStyle: 'italic',
                    color: 'var(--color-orange)',
                    fontWeight: 300,
                    fontVariationSettings: '"opsz" 60, "SOFT" 100',
                  }}
                >
                  chantier
                </em>
              </h2>
            </div>
          </div>
          <p className="text-[14px] text-ink-2 mb-8 -mt-2">
            Déclarez un nouveau site avec sa deadline. L'IA optimisera le planning selon la météo locale.
          </p>

          <DashboardNewSiteForm
            canCreate={canCreate}
            limitReason={limitReason}
            currentCount={currentCount}
            maxSites={limits.maxSites}
          />
        </div>

        {/* ── COL DROITE : chantiers + deadlines ── */}
        <div className="p-8 lg:p-12">

          {/* En-tête section */}
          <div className="flex items-end justify-between mb-8 pb-5 border-b border-rule-soft">
            <div>
              <div className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-2">
                Section 02 · Activité
              </div>
              <h2
                className="font-serif text-[28px] leading-none tracking-tight text-ink lg:text-[32px]"
                style={{ fontVariationSettings: '"opsz" 60, "SOFT" 30' }}
              >
                Occupation des chantiers
              </h2>
            </div>
            {sitesArray.length > 0 && (
              <Link
                href="/sites"
                className="font-mono text-[11px] uppercase tracking-widest text-ink-2 border-b border-ink-2 pb-0.5 hover:text-orange hover:border-orange transition-colors flex-shrink-0 ml-4"
              >
                Voir tout →
              </Link>
            )}
          </div>
          <p className="text-[14px] text-ink-2 mb-6 -mt-2">Plannings et taux d'occupation en cours.</p>

          {/* Liste des chantiers */}
          {sitesArray.length === 0 ? (
            <div className="border border-dashed border-rule-soft p-8 text-center">
              <p className="font-mono text-[11px] uppercase tracking-widest text-ink-3">
                Aucun chantier · Créez-en un depuis le formulaire
              </p>
            </div>
          ) : (
            <div className="border-t border-rule-soft">
              {activeSites.slice(0, 5).map((site, i) => {
                const stats = siteStats[site.id] || { tasks: 0, done: 0, workers: 0 };
                const progress = stats.tasks > 0 ? Math.round((stats.done / stats.tasks) * 100) : 0;
                const isOverdue = site.deadline && new Date(site.deadline) < new Date() && !site.completed_at;
                const loc = site.postal_code || site.address || null;

                return (
                  <Link
                    key={site.id}
                    href={`/site/${site.id}/dashboard`}
                    className="group block py-6 border-b border-rule-soft hover:bg-paper-2 -mx-8 px-8 lg:-mx-12 lg:px-12 transition-colors duration-150 relative"
                  >
                    {/* Bordure gauche orange au hover */}
                    <div className="absolute left-0 top-0 bottom-0 w-[3px] bg-transparent group-hover:bg-orange transition-colors duration-150" />

                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="font-mono text-[11px] text-ink-3 tracking-widest mb-1.5">
                          N° {fmt2(i + 1)} · CHANTIER
                        </div>
                        <div
                          className="font-serif text-[20px] text-ink mb-3 leading-tight"
                          style={{ fontVariationSettings: '"opsz" 60, "SOFT" 30' }}
                        >
                          {site.name}
                        </div>
                        <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-ink-2 tracking-[0.05em] mb-3">
                          {loc && <span>⌖ {loc}</span>}
                          {site.deadline && (
                            <span>⌚ Livr. {new Date(site.deadline).toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', year: '2-digit' })}</span>
                          )}
                          {stats.workers > 0 && <span>⚆ {stats.workers} employé{stats.workers > 1 ? 's' : ''}</span>}
                        </div>
                        {/* Progress bar orange */}
                        <div className="flex items-center gap-3 max-w-[300px]">
                          <div className="flex-1 h-[4px] bg-paper-2 relative overflow-hidden">
                            <div
                              className="absolute inset-y-0 left-0 bg-orange"
                              style={{ width: `${progress}%` }}
                            />
                          </div>
                          <span className="font-mono text-[12px] text-ink min-w-[36px]">{progress} %</span>
                        </div>
                      </div>
                      <div className="flex-shrink-0 flex flex-col items-end gap-2">
                        <span
                          className={`inline-flex items-center gap-1.5 px-2 py-1 border font-mono text-[10px] uppercase tracking-widest ${
                            isOverdue
                              ? 'border-orange text-orange'
                              : 'border-rule-soft text-ink-2'
                          }`}
                        >
                          <span
                            className={`w-1.5 h-1.5 rounded-full ${isOverdue ? 'bg-orange' : 'bg-green'}`}
                          />
                          {isOverdue ? 'Retard' : 'Actif'}
                        </span>
                        <span className="font-mono text-[18px] text-ink-3 group-hover:text-orange transition-colors mt-8">↗</span>
                      </div>
                    </div>
                  </Link>
                );
              })}

              {/* Chantiers terminés (résumé) */}
              {completedSites.length > 0 && (
                <div className="py-4 border-b border-rule-soft">
                  <span className="font-mono text-[10px] uppercase tracking-widest text-ink-3">
                    + {completedSites.length} chantier{completedSites.length > 1 ? 's' : ''} terminé{completedSites.length > 1 ? 's' : ''} ·{' '}
                    <Link href="/sites" className="underline hover:text-orange transition-colors">Voir tout</Link>
                  </span>
                </div>
              )}
            </div>
          )}

          {/* ── Prochaines deadlines ── */}
          {upcomingDeadlines.length > 0 && (
            <div className="mt-10">
              <div className="flex items-end justify-between mb-3 pb-3 border-b border-rule-soft">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-1">
                    Section 03 · Échéances
                  </div>
                  <h3
                    className="font-serif text-[22px] leading-none tracking-tight text-ink"
                    style={{ fontVariationSettings: '"opsz" 60, "SOFT" 30' }}
                  >
                    Prochaines deadlines
                  </h3>
                </div>
              </div>

              {upcomingDeadlines.map((site) => {
                const d = new Date(site.deadline!);
                const day = d.getDate();
                const month = d.toLocaleDateString('fr-FR', { month: 'long' });
                const year = String(d.getFullYear()).slice(2);
                const remaining = daysUntil(site.deadline!);
                const loc = site.postal_code || site.address || null;

                return (
                  <Link
                    key={site.id}
                    href={`/site/${site.id}/dashboard`}
                    className="grid gap-5 py-5 border-b border-rule-soft hover:bg-paper-2 -mx-8 px-8 lg:-mx-12 lg:px-12 transition-colors duration-150"
                    style={{ gridTemplateColumns: '60px 1fr auto', alignItems: 'center' }}
                  >
                    {/* Jour */}
                    <div className="text-right">
                      <div
                        className="font-serif text-[36px] leading-none tracking-tight text-ink"
                        style={{ fontVariationSettings: '"opsz" 60, "SOFT" 30', letterSpacing: '-0.03em' }}
                      >
                        {day}
                      </div>
                      <div className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mt-1">
                        {month} {year}
                      </div>
                    </div>

                    {/* Nom + lieu */}
                    <div>
                      <div className="font-sans font-medium text-[15px] text-ink mb-0.5">{site.name}</div>
                      {loc && (
                        <div className="font-mono text-[11px] text-ink-2 tracking-[0.05em]">{loc}</div>
                      )}
                    </div>

                    {/* Compteur */}
                    <div
                      className={`font-mono text-[11px] uppercase tracking-widest ${
                        remaining < 0 ? 'text-danger' : remaining <= 7 ? 'text-warn' : 'text-orange'
                      }`}
                    >
                      {remaining < 0 ? `J + ${Math.abs(remaining)}` : `J − ${remaining}`}
                    </div>
                  </Link>
                );
              })}
            </div>
          )}

        </div>
      </section>
    </div>
  );
}
