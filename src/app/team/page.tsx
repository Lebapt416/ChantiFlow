import { redirect } from 'next/navigation';
import Link from 'next/link';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AddWorkerForm } from './add-worker-form';
import { AddWorkerToSiteForm } from './add-worker-to-site-form';
import { DeleteWorkerButton } from '@/components/delete-worker-button';
import { WorkerConnectionQrButton } from '@/components/worker-connection-qr-button';
import { CopyButton } from '@/components/copy-button';

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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let accountWorkers: any[] = [];

  try {
    const { error: testError } = await supabase
      .from('workers')
      .select('id, name, email, role, created_at, status, access_token, access_code, site_id')
      .eq('created_by', user.id)
      .is('site_id', null)
      .limit(1);

    if (testError) {
      if (testError.message.includes('created_by') || testError.message.includes('column') || testError.code === '42703') {
        console.warn('Colonne created_by ou status non trouvée - migration non exécutée');
        const { data: workersWithoutStatus } = await supabase
          .from('workers')
          .select('id, name, email, role, created_at, access_token, access_code, site_id')
          .eq('created_by', user.id)
          .is('site_id', null)
          .order('created_at', { ascending: true });
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        accountWorkers = (workersWithoutStatus ?? []).map((w: any) => ({ ...w, status: null }));
      } else {
        throw testError;
      }
    } else {
      const { data: allAccountWorkers, error: fetchError } = await supabase
        .from('workers')
        .select('id, name, email, role, created_at, status, access_token, access_code, site_id')
        .eq('created_by', user.id)
        .is('site_id', null)
        .order('created_at', { ascending: true });

      if (fetchError && fetchError.message.includes('status')) {
        const { data: workersWithoutStatus } = await supabase
          .from('workers')
          .select('id, name, email, role, created_at, access_token, access_code, site_id')
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
    console.warn('Erreur récupération workers compte:', error?.message);
    accountWorkers = [];
  }

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
        .select('id, name, email, role, site_id, created_at, access_token, access_code')
        .in('site_id', siteIds)
        .order('created_at', { ascending: true })
    : { data: [] };

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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pendingWorkers = (accountWorkers || []).filter((w: any) => w.status === 'pending');
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const approvedWorkers = (accountWorkers || []).filter((w: any) =>
    w.status === 'approved' || w.status === null || w.status === undefined
  );

  const activeSites = sites?.filter((site) => !site.completed_at) ?? [];

  return (
    <AppShell
      heading="Équipe générale"
      subheading="Gérez votre catalogue d'équipe et ajoutez rapidement des personnes."
      userEmail={user.email}
      primarySite={sites?.[0] ?? null}
    >
      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded border border-rule-soft bg-paper p-5 dark:border-rule dark:bg-ink">
          <p className="text-xs uppercase tracking-[0.3em] text-ink-3">Membres</p>
          <p className="mt-2 text-3xl font-semibold text-ink dark:text-paper">{workers?.length ?? 0}</p>
          <p className="text-sm text-ink-3">collaborateurs actifs</p>
        </div>
        <div className="rounded border border-rule-soft bg-paper p-5 dark:border-rule dark:bg-ink">
          <p className="text-xs uppercase tracking-[0.3em] text-ink-3">Chantiers</p>
          <p className="mt-2 text-3xl font-semibold text-ink dark:text-paper">{sites?.length ?? 0}</p>
          <p className="text-sm text-ink-3">sites assignés</p>
        </div>
        <div className="rounded border border-rule-soft bg-paper p-5 dark:border-rule dark:bg-ink">
          <p className="text-xs uppercase tracking-[0.3em] text-ink-3">Rôles</p>
          <p className="mt-2 text-3xl font-semibold text-ink dark:text-paper">{Object.keys(groupedByRole).length}</p>
          <p className="text-sm text-ink-3">profils différents</p>
        </div>
      </section>

      <section className="mt-8 rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
        <h2 className="font-serif text-[22px] text-ink dark:text-paper">
          Répartition des métiers
        </h2>
        {highlightedRoles.length ? (
          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {highlightedRoles.map(([role, count]) => (
              <div
                key={role}
                className="rounded border border-rule-soft p-4 text-sm dark:border-rule"
              >
                <p className="text-xs uppercase tracking-[0.3em] text-ink-3">
                  {role}
                </p>
                <p className="mt-2 text-2xl font-semibold text-ink dark:text-paper">
                  {count}
                </p>
                <p className="text-xs text-ink-3">personnes</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-4 text-sm text-ink-3">
            Aucun employé renseigné.
          </p>
        )}
      </section>

      <section className="mt-8 rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="font-serif text-[22px] text-ink dark:text-paper">
              QR code d&apos;inscription à l&apos;équipe
            </h2>
            <p className="text-sm text-ink-3">
              Partagez ce QR code pour permettre aux personnes de s&apos;ajouter à votre équipe en le scannant.
            </p>
          </div>
        </div>
        <div className="mb-8">
          <Link
            href="/team/qr"
            className="inline-flex items-center justify-center gap-2 border border-orange px-6 py-3 font-mono text-[11px] uppercase tracking-widest text-orange transition hover:bg-paper-2"
          >
            Voir le QR code d&apos;inscription
          </Link>
        </div>
      </section>

      <section className="mt-8 rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="font-serif text-[22px] text-ink dark:text-paper">
              Ajouter rapidement un membre au catalogue
            </h2>
            <p className="text-sm text-ink-3">
              Ajoutez un nouveau membre à votre catalogue d&apos;équipe. Il sera disponible pour être assigné à n&apos;importe quel chantier.
            </p>
          </div>
        </div>
        <div className="mb-8">
          <AddWorkerForm />
        </div>
      </section>

      {activeSites.length > 0 && (
        <section className="mt-8 rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h2 className="font-serif text-[22px] text-ink dark:text-paper">
                Ajouter un membre à un chantier
              </h2>
              <p className="text-sm text-ink-3">
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

      <section className="mt-8 rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="font-serif text-[22px] text-ink dark:text-paper">
              Listing complet
            </h2>
            <p className="text-sm text-ink-3">
              Filtre par site via les boutons rapides.
            </p>
          </div>
        </div>
        {(() => {
          return (
            <>
              {pendingWorkers.length > 0 && (
                <div className="space-y-3 mb-6">
                  <h3 className="font-mono text-[11px] uppercase tracking-widest text-warn mb-2">
                    ⏳ Demandes en attente de validation ({pendingWorkers.length})
                  </h3>
                  {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                  {pendingWorkers.map((worker: any) => (
                    <div
                      key={worker.id}
                      className="rounded border border-warn bg-paper-2 p-4"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <p className="text-sm font-semibold text-ink dark:text-paper">
                              {worker.name}
                            </p>
                            {worker.access_code && (
                              <span className="inline-flex items-center gap-1 border border-rule-soft bg-paper-2 px-2 py-0.5 font-mono text-[10px] text-ink">
                                {worker.access_code}
                                <CopyButton value={worker.access_code} />
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-ink-3">
                            {worker.role ?? 'Rôle non défini'}
                          </p>
                          <p className="text-xs text-ink-3">
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
                              className="border border-orange px-3 py-1.5 font-mono text-[10px] uppercase tracking-widest text-orange transition hover:bg-paper-2"
                            >
                              Valider
                            </button>
                          </form>
                          <form action="/team/actions" method="post" className="inline">
                            <input type="hidden" name="action" value="reject" />
                            <input type="hidden" name="workerId" value={worker.id} />
                            <button
                              type="submit"
                              className="border border-danger px-3 py-1.5 font-mono text-[10px] uppercase tracking-widest text-danger transition hover:bg-paper-2"
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
                  <h3 className="font-mono text-[11px] uppercase tracking-widest text-ink-2 mb-2">
                    Membres de votre équipe (disponibles pour tous les chantiers)
                  </h3>
                  {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                  {approvedWorkers.map((worker: any) => (
                    <div
                      key={worker.id}
                      className="rounded border border-rule-soft bg-paper-2 p-4 dark:border-rule"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <p className="text-sm font-semibold text-ink dark:text-paper">
                              {worker.name}
                            </p>
                            {worker.access_code && (
                              <span className="inline-flex items-center gap-1 border border-rule-soft bg-paper px-2 py-0.5 font-mono text-[10px] text-ink">
                                {worker.access_code}
                                <CopyButton value={worker.access_code} />
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-ink-3">
                            {worker.role ?? 'Rôle non défini'}
                          </p>
                          <p className="text-xs text-ink-3">
                            {worker.email ?? 'Email non communiqué'}
                          </p>
                        </div>
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-mono text-[10px] uppercase tracking-widest text-orange dark:text-green">
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
                <p className="text-sm text-ink-3">
                  Aucune ressource pour l&apos;instant. Ajoutez des membres à votre équipe ci-dessus.
                </p>
              )}
            </>
          );
        })()}
        {siteWorkers && siteWorkers.length > 0 ? (
          <div className="space-y-3 mt-6">
            <h3 className="font-mono text-[11px] uppercase tracking-widest text-ink-2 mb-2">
              Membres assignés à des chantiers spécifiques
            </h3>
            {siteWorkers.map((worker) => (
              <div
                key={worker.id}
                className="rounded border border-rule-soft p-4 dark:border-rule"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-semibold text-ink dark:text-paper">
                        {worker.name}
                      </p>
                      {worker.access_code && (
                        <span className="inline-flex items-center gap-1 border border-rule-soft bg-paper-2 px-2 py-0.5 font-mono text-[10px] text-ink">
                          {worker.access_code}
                          <CopyButton value={worker.access_code} />
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-ink-3">
                      {worker.role ?? 'Rôle non défini'}
                    </p>
                    <p className="text-xs text-ink-3">
                      {worker.email ?? 'Email non communiqué'}
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    {worker.site_id ? (
                      <Link
                        href={`/site/${worker.site_id}`}
                        className="font-mono text-[10px] uppercase tracking-widest text-ink-2 hover:text-ink"
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
          <p className="text-sm text-ink-3">
            Aucune ressource pour l&apos;instant. Ajoutez des membres à votre équipe ci-dessus.
          </p>
        ) : null}
      </section>
    </AppShell>
  );
}
