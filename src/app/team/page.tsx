import { redirect } from 'next/navigation';
import Link from 'next/link';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AddWorkerForm } from './add-worker-form';
import { AddWorkerToSiteForm } from './add-worker-to-site-form';
import { DeleteWorkerButton } from '@/components/delete-worker-button';
import { WorkerConnectionQrButton } from '@/components/worker-connection-qr-button';

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
  const appBaseUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? 'http://localhost:3000';

  // Récupérer les workers au niveau du compte (sans site_id)
  // Gérer le cas où created_by n'existe pas encore (migration non exécutée)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let accountWorkers: any[] = [];
  
  // Vérifier d'abord si la colonne created_by existe en essayant une requête simple
  // Si elle échoue, on sait que la migration n'a pas été exécutée
  try {
    // Essayer de récupérer les workers avec created_by
    // Tester d'abord si la colonne status existe
    const { error: testError } = await supabase
      .from('workers')
      .select('id, name, email, role, created_at, status, access_token, site_id')
      .eq('created_by', user.id)
      .is('site_id', null)
      .limit(1);
    
    if (testError) {
      // Si l'erreur mentionne created_by ou column, la migration n'est pas exécutée
      if (testError.message.includes('created_by') || testError.message.includes('column') || testError.code === '42703') {
        console.warn('Colonne created_by ou status non trouvée - migration non exécutée');
        // Essayer sans status
        const { data: workersWithoutStatus } = await supabase
          .from('workers')
          .select('id, name, email, role, created_at, access_token, site_id')
          .eq('created_by', user.id)
          .is('site_id', null)
          .order('created_at', { ascending: true });
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        accountWorkers = (workersWithoutStatus ?? []).map((w: any) => ({ ...w, status: null }));
      } else {
        // Autre erreur, on la propage
        throw testError;
      }
    } else {
      // Si pas d'erreur, récupérer tous les workers avec status
      const { data: allAccountWorkers, error: fetchError } = await supabase
        .from('workers')
        .select('id, name, email, role, created_at, status, access_token, site_id')
        .eq('created_by', user.id)
        .is('site_id', null)
        .order('created_at', { ascending: true });
      
      if (fetchError && fetchError.message.includes('status')) {
        // Si la colonne status n'existe pas, récupérer sans status
        const { data: workersWithoutStatus } = await supabase
          .from('workers')
          .select('id, name, email, role, created_at, access_token, site_id')
          .eq('created_by', user.id)
          .is('site_id', null)
          .order('created_at', { ascending: true });
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        accountWorkers = (workersWithoutStatus ?? []).map((w: any) => ({ ...w, status: null }));
      } else {
        accountWorkers = allAccountWorkers ?? [];
      }
    }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } catch (error: any) {
    // Colonne created_by n'existe pas encore, on continue avec un tableau vide
    console.warn('Erreur récupération workers compte:', error?.message);
    accountWorkers = [];
  }

  // Récupérer aussi les workers liés aux chantiers pour affichage
  const { data: sites } = await supabase
    .from('sites')
    .select('id, name, completed_at')
    .eq('created_by', user.id)
    .order('created_at', { ascending: false });

  const siteMap =
    sites?.reduce<Record<string, string>>((acc, site) => {
      acc[site.id] = site.name;
      return acc;
    }, {}) ?? {};

  const siteIds = Object.keys(siteMap);

  const { data: siteWorkers } = siteIds.length
    ? await supabase
        .from('workers')
        .select('id, name, email, role, site_id, created_at, access_token')
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

  // Filtrer les workers en attente et approuvés
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pendingWorkers = (accountWorkers || []).filter((w: any) => w.status === 'pending');
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const approvedWorkers = (accountWorkers || []).filter((w: any) => 
    w.status === 'approved' || w.status === null || w.status === undefined
  );

  // Filtrer uniquement les chantiers actifs (non terminés)
  const activeSites = sites?.filter((site) => !site.completed_at) ?? [];

  return (
    <AppShell
      heading="Équipe générale"
      subheading="Gérez votre catalogue d'équipe et ajoutez rapidement des personnes."
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
            QR code d&apos;inscription à l&apos;équipe
          </h2>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Partagez ce QR code pour permettre aux personnes de s&apos;ajouter à votre équipe en le scannant.
          </p>
        </div>
      </div>
      <div className="mb-8">
        <Link
          href="/team/qr"
          className="flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-emerald-700 active:scale-95"
        >
          <span>Voir le QR code d&apos;inscription</span>
        </Link>
      </div>
    </section>

    <section className="mt-8 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
            Ajouter rapidement un membre au catalogue
          </h2>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Ajoutez un nouveau membre à votre catalogue d&apos;équipe. Il sera disponible pour être assigné à n&apos;importe quel chantier.
          </p>
        </div>
      </div>
      <div className="mb-8">
        <AddWorkerForm />
      </div>
    </section>

    {activeSites.length > 0 && (
      <section className="mt-8 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Ajouter un membre à un chantier
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Sélectionnez un chantier et ajoutez un membre de votre équipe ou créez un nouveau membre directement pour ce chantier.
            </p>
          </div>
        </div>
        <div className="mb-8">
          <AddWorkerToSiteForm 
            sites={activeSites} 
            availableWorkers={approvedWorkers} 
          />
        </div>
      </section>
    )}

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
      {/* Séparer les workers en attente et approuvés */}
      {(() => {
        return (
          <>
            {pendingWorkers.length > 0 && (
              <div className="space-y-3 mb-6">
                <h3 className="text-sm font-semibold text-amber-700 dark:text-amber-300 mb-2">
                  ⏳ Demandes en attente de validation ({pendingWorkers.length})
                </h3>
                {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                {pendingWorkers.map((worker: any) => (
                  <div
                    key={worker.id}
                    className="rounded-2xl border border-amber-200 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-900/20"
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
                      <div className="flex flex-wrap items-center gap-2">
                        <WorkerConnectionQrButton
                          token={worker.access_token}
                          workerName={worker.name}
                          baseUrl={appBaseUrl}
                        />
                        <form action="/team/actions" method="post" className="inline">
                          <input type="hidden" name="action" value="approve" />
                          <input type="hidden" name="workerId" value={worker.id} />
                          <button
                            type="submit"
                            className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-emerald-700"
                          >
                            Valider
                          </button>
                        </form>
                        <form action="/team/actions" method="post" className="inline">
                          <input type="hidden" name="action" value="reject" />
                          <input type="hidden" name="workerId" value={worker.id} />
                          <button
                            type="submit"
                            className="rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-rose-700"
                          >
                            Refuser
                          </button>
                        </form>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
            
            {approvedWorkers.length > 0 && (
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-2">
                  Membres de votre équipe (disponibles pour tous les chantiers)
                </h3>
                {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                {approvedWorkers.map((worker: any) => (
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
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                          Disponible
                        </span>
                        <WorkerConnectionQrButton
                          token={worker.access_token}
                          workerName={worker.name}
                          baseUrl={appBaseUrl}
                        />
                        <DeleteWorkerButton workerId={worker.id} workerName={worker.name} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {pendingWorkers.length === 0 && approvedWorkers.length === 0 && (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Aucune ressource pour l&apos;instant. Ajoutez des membres à votre équipe ci-dessus.
              </p>
            )}
          </>
        );
      })()}
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
                <div className="flex flex-wrap items-center gap-2">
                  {worker.site_id ? (
                    <Link
                      href={`/site/${worker.site_id}`}
                      className="text-xs font-semibold text-black hover:underline dark:text-white"
                    >
                      {siteMap[worker.site_id] ?? 'Site inconnu'} →
                    </Link>
                  ) : null}
                  <WorkerConnectionQrButton
                    token={worker.access_token}
                    workerName={worker.name}
                    baseUrl={appBaseUrl}
                  />
                  <DeleteWorkerButton workerId={worker.id} workerName={worker.name} />
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : null}
      {(!accountWorkers || accountWorkers.length === 0) && (!siteWorkers || siteWorkers.length === 0) ? (
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Aucune ressource pour l&apos;instant. Ajoutez des membres à votre équipe ci-dessus.
        </p>
      ) : null}
    </section>
    </AppShell>
  );
}

