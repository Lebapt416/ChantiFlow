import { redirect, notFound } from 'next/navigation';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AddWorkerForm } from '../add-worker-form';
import { DeleteWorkerButton } from '@/components/delete-worker-button';
import { CopyButton } from '@/components/copy-button';

type Params = {
  params: Promise<{
    id: string;
  }>;
};

type TeamMember = {
  id: string;
  name: string | null;
  email: string | null;
  role: string | null;
  status?: 'approved' | 'pending' | 'rejected' | null;
  access_code?: string | null;
  created_at?: string | null;
};

type SelectableWorker = {
  id: string;
  name: string;
  email: string | null;
  role: string | null;
};

function isValidUuid(value: string | undefined) {
  return Boolean(value && /^[0-9a-fA-F-]{36}$/.test(value));
}

export default async function SiteTeamPage({ params }: Params) {
  const { id } = await params;

  if (!isValidUuid(id)) {
    notFound();
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  // Récupérer les infos du chantier
  const { data: site, error: siteError } = await supabase
    .from('sites')
    .select('id, name, deadline, created_by, completed_at')
    .eq('id', id)
    .eq('created_by', user.id)
    .single();

  if (siteError || !site) {
    notFound();
  }

  // Récupérer les workers assignés à ce chantier avec le code d'accès
  const { data: siteWorkersData } = await supabase
    .from('workers')
    .select('id, name, email, role, created_at, access_code, status')
    .eq('site_id', id)
    .order('created_at', { ascending: false });
  
  // Log pour debug
  console.log('[SiteTeamPage] Workers récupérés:', siteWorkersData?.map(w => ({ 
    name: w.name, 
    email: w.email, 
    access_code: w.access_code 
  })));
  const siteWorkers: TeamMember[] = (siteWorkersData ?? []).map((worker) => ({
    ...worker,
  }));

  // Récupérer les workers disponibles au niveau du compte (pour le formulaire)
  let availableWorkers: SelectableWorker[] = [];
  try {
    const { data: accountWorkersData } = await supabase
      .from('workers')
      .select('id, name, email, role, created_at, status')
      .eq('created_by', user.id)
      .is('site_id', null)
      .in('status', ['approved'])
      .order('created_at', { ascending: false });

    availableWorkers = (accountWorkersData ?? []).map((worker) => ({
      id: worker.id,
      name: worker.name ?? 'Membre sans nom',
      email: worker.email,
      role: worker.role,
    }));
  } catch (error) {
    console.warn('Erreur récupération workers compte:', error);
  }

  // Calculer les stats
  const totalWorkers = siteWorkers.length;
  const workersWithEmail = siteWorkers.filter((w) => w.email).length;
  const groupedByRole = siteWorkers.reduce<Record<string, number>>((acc, worker) => {
    const key = worker.role?.toLowerCase() || 'Non défini';
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});

  const isCompleted = !!site.completed_at;

  return (
    <AppShell
      heading={`Équipe - ${site.name}`}
      subheading="Gérez les membres de votre équipe pour ce chantier"
      userEmail={user.email}
      primarySite={site}
    >
      {isCompleted && (
        <div className="mb-6 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-200">
          ⚠️ Ce chantier est terminé. Vous ne pouvez plus ajouter de membres.
        </div>
      )}

      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Membres</p>
          <p className="mt-2 text-3xl font-semibold">{totalWorkers}</p>
          <p className="text-sm text-zinc-500">sur ce chantier</p>
        </div>
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Avec email</p>
          <p className="mt-2 text-3xl font-semibold">{workersWithEmail}</p>
          <p className="text-sm text-zinc-500">membres contactables</p>
        </div>
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Rôles</p>
          <p className="mt-2 text-3xl font-semibold">{Object.keys(groupedByRole).length}</p>
          <p className="text-sm text-zinc-500">profils différents</p>
        </div>
      </section>

      {Object.keys(groupedByRole).length > 0 && (
        <section className="mt-8 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white mb-4">
            Répartition des métiers
          </h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {Object.entries(groupedByRole).map(([role, count]) => (
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
        </section>
      )}

      {!isCompleted && (
        <section className="mt-8 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
                Ajouter un membre au chantier
              </h2>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Ajoutez un membre de votre équipe ou créez un nouveau membre pour ce chantier.
              </p>
            </div>
          </div>
          <div className="mb-8">
            <AddWorkerForm siteId={id} availableWorkers={availableWorkers} />
          </div>
        </section>
      )}

      <section className="mt-8 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Membres du chantier
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Liste de tous les membres assignés à ce chantier.
            </p>
          </div>
        </div>

        {siteWorkers.length > 0 ? (
          <div className="space-y-3">
            {siteWorkers.map((worker) => (
              <div
                key={worker.id}
                className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-900/50"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                      {worker.name}
                    </p>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      {worker.role ?? 'Rôle non défini'}
                    </p>
                    {worker.email && (
                      <p className="text-xs text-zinc-400 dark:text-zinc-500">
                        {worker.email}
                      </p>
                    )}
                    {worker.access_code ? (
                      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                        <span className="text-zinc-600 dark:text-zinc-400">Code d&apos;accès :</span>
                        <span className="font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                          {worker.access_code}
                        </span>
                        <CopyButton value={worker.access_code} />
                        <span className="text-zinc-500 dark:text-zinc-500 text-[10px]">
                          (envoyé par email)
                        </span>
                      </div>
                    ) : (
                      <div className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
                        Code d&apos;accès : <span className="italic">Non généré (sera créé lors de l&apos;assignation d&apos;une tâche)</span>
                      </div>
                    )}
                  </div>
                  {!isCompleted && (
                    <div className="flex items-center gap-2">
                      <DeleteWorkerButton
                        workerId={worker.id}
                        workerName={worker.name ?? 'Membre sans nom'}
                      />
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Aucun membre assigné à ce chantier pour le moment.
          </p>
        )}
      </section>
    </AppShell>
  );
}

