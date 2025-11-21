import { redirect } from 'next/navigation';
import Link from 'next/link';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AddWorkerForm } from './add-worker-form';
import { DeleteWorkerButton } from '@/components/delete-worker-button';

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

  // Récupérer les workers au niveau du compte (sans site_id)
  // Gérer le cas où created_by n'existe pas encore (migration non exécutée)
  let accountWorkers: any[] = [];
  
  // Vérifier d'abord si la colonne created_by existe en essayant une requête simple
  // Si elle échoue, on sait que la migration n'a pas été exécutée
  try {
    // Essayer de récupérer les workers avec created_by
    const { data, error } = await supabase
      .from('workers')
      .select('id, name, email, role, created_at')
      .eq('created_by', user.id)
      .is('site_id', null)
      .order('created_at', { ascending: true })
      .limit(1); // Limiter pour tester rapidement
    
    if (error) {
      // Si l'erreur mentionne created_by ou column, la migration n'est pas exécutée
      if (error.message.includes('created_by') || error.message.includes('column') || error.code === '42703') {
        console.warn('Colonne created_by non trouvée - migration non exécutée');
        accountWorkers = [];
      } else {
        // Autre erreur, on la propage
        throw error;
      }
    } else {
      // Si pas d'erreur, récupérer tous les workers
      const { data: allAccountWorkers } = await supabase
        .from('workers')
        .select('id, name, email, role, created_at')
        .eq('created_by', user.id)
        .is('site_id', null)
        .order('created_at', { ascending: true });
      
      accountWorkers = allAccountWorkers ?? [];
    }
  } catch (error: any) {
    // Colonne created_by n'existe pas encore, on continue avec un tableau vide
    console.warn('Erreur récupération workers compte:', error?.message);
    accountWorkers = [];
  }

  // Récupérer aussi les workers liés aux chantiers pour affichage
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

  const { data: siteWorkers } = siteIds.length
    ? await supabase
        .from('workers')
        .select('id, name, email, role, site_id, created_at')
        .in('site_id', siteIds)
        .order('created_at', { ascending: true })
    : { data: [] };

  // Combiner les workers du compte et des chantiers pour les stats
  const workers = [...(accountWorkers ?? []), ...(siteWorkers ?? [])];

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
            Ajouter un membre
          </h2>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Ajoutez un nouveau membre à votre équipe.
          </p>
        </div>
      </div>
      <div className="mb-8">
        <AddWorkerForm />
      </div>
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
      {accountWorkers && accountWorkers.length > 0 ? (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-2">
            Membres de votre équipe (disponibles pour tous les chantiers)
          </h3>
          {accountWorkers.map((worker) => (
            <div
              key={worker.id}
              className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 dark:border-emerald-800 dark:bg-emerald-900/20"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex-1">
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
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                    Disponible
                  </span>
                  <DeleteWorkerButton workerId={worker.id} workerName={worker.name} />
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : null}
      {siteWorkers && siteWorkers.length > 0 ? (
        <div className="space-y-3 mt-6">
          <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-2">
            Membres assignés à des chantiers spécifiques
          </h3>
          {siteWorkers.map((worker) => (
            <div
              key={worker.id}
              className="rounded-2xl border border-zinc-200 p-4 dark:border-zinc-700"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex-1">
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
                <div className="flex items-center gap-2">
                  {worker.site_id ? (
                    <Link
                      href={`/site/${worker.site_id}`}
                      className="text-xs font-semibold text-black hover:underline dark:text-white"
                    >
                      {siteMap[worker.site_id] ?? 'Site inconnu'} →
                    </Link>
                  ) : null}
                  <DeleteWorkerButton workerId={worker.id} workerName={worker.name} />
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : null}
      {(!accountWorkers || accountWorkers.length === 0) && (!siteWorkers || siteWorkers.length === 0) ? (
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Aucune ressource pour l'instant. Ajoutez des membres à votre équipe ci-dessus.
        </p>
      ) : null}
    </section>
    </AppShell>
  );
}

