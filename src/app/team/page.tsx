import { redirect } from 'next/navigation';
import Link from 'next/link';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export const metadata = {
  title: 'Équipe | ChantiFlow',
};

export default async function TeamPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: sites } = await supabase
    .from('sites')
    .select('id, name')
    .eq('created_by', user.id);

  const siteMap =
    sites?.reduce<Record<string, string>>((acc, site) => {
      acc[site.id] = site.name;
      return acc;
    }, {}) ?? {};

  const siteIds = Object.keys(siteMap);

  const { data: workers } = siteIds.length
    ? await supabase
        .from('workers')
        .select('id, name, email, role, site_id, created_at')
        .in('site_id', siteIds)
        .order('created_at', { ascending: true })
    : { data: [] };

  const groupedByRole =
    workers?.reduce<Record<string, number>>((acc, worker) => {
      const key = worker.role?.toLowerCase() || 'Non défini';
      acc[key] = (acc[key] ?? 0) + 1;
      return acc;
    }, {}) ?? {};

  const highlightedRoles = Object.entries(groupedByRole)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);

  return (
    <AppShell
      heading="Équipe"
      subheading="Centralise les intervenants présents sur tes chantiers."
      userEmail={user.email}
      primarySite={sites?.[0] ?? null}
    >
      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Membres</p>
          <p className="mt-2 text-3xl font-semibold">{workers?.length ?? 0}</p>
          <p className="text-sm text-zinc-500">collaborateurs actifs</p>
        </div>
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Chantiers</p>
          <p className="mt-2 text-3xl font-semibold">{sites?.length ?? 0}</p>
          <p className="text-sm text-zinc-500">sites assignés</p>
        </div>
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Rôles</p>
          <p className="mt-2 text-3xl font-semibold">{Object.keys(groupedByRole).length}</p>
          <p className="text-sm text-zinc-500">profils différents</p>
        </div>
      </section>

      <section className="mt-8 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
        Répartition des métiers
      </h2>
      {highlightedRoles.length ? (
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {highlightedRoles.map(([role, count]) => (
            <div
              key={role}
              className="rounded-2xl border border-zinc-200 p-4 text-sm dark:border-zinc-700"
            >
              <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
                {role}
              </p>
              <p className="mt-2 text-2xl font-semibold text-zinc-900 dark:text-white">
                {count}
              </p>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">personnes</p>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-4 text-sm text-zinc-500 dark:text-zinc-400">
          Aucun employé renseigné.
        </p>
      )}
    </section>

    <section className="mt-8 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
            Listing complet
          </h2>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Filtre par site via les boutons rapides.
          </p>
        </div>
      </div>
      {workers?.length ? (
        <div className="space-y-3">
          {workers.map((worker) => (
            <div
              key={worker.id}
              className="rounded-2xl border border-zinc-200 p-4 dark:border-zinc-700"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                    {worker.name}
                  </p>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    {worker.role ?? 'Rôle non défini'}
                  </p>
                  <p className="text-xs text-zinc-400 dark:text-zinc-500">
                    {worker.email ?? 'Email non communiqué'}
                  </p>
                </div>
                <Link
                  href={`/site/${worker.site_id}`}
                  className="text-xs font-semibold text-black hover:underline dark:text-white"
                >
                  {siteMap[worker.site_id] ?? 'Site inconnu'} →
                </Link>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Aucune ressource pour l’instant. Ajoute des employés depuis une fiche chantier.
        </p>
      )}
    </section>
    </AppShell>
  );
}

