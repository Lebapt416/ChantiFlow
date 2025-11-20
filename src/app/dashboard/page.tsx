import Link from 'next/link';
import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { CreateSiteForm } from './create-site-form';
import { signOutAction } from '../actions';
import { AppShell } from '@/components/app-shell';

export const metadata = {
  title: 'Dashboard | ChantiFlow',
};

export default async function DashboardPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: sites, error } = await supabase
    .from('sites')
    .select('id, name, deadline, created_at')
    .eq('created_by', user.id)
    .order('created_at', { ascending: false });

  const siteIds = sites?.map((site) => site.id) ?? [];

  let totalTasks = 0;
  let doneTasks = 0;

  if (siteIds.length) {
    const { data: tasks } = await supabase
      .from('tasks')
      .select('status')
      .in('site_id', siteIds);

    totalTasks = tasks?.length ?? 0;
    doneTasks = tasks?.filter((task) => task.status === 'done').length ?? 0;
  }

  const pendingTasks = totalTasks - doneTasks;

  const nextDeadlines = (sites ?? [])
    .filter((site) => site.deadline)
    .sort((a, b) => (a.deadline ?? '').localeCompare(b.deadline ?? ''))
    .slice(0, 4);

  return (
    <AppShell
      heading="Mes chantiers"
      subheading="Crée, supervise et partage l’avancement de tes sites."
      userEmail={user.email}
      primarySite={sites?.[0] ?? null}
      actions={
        <form action={signOutAction}>
          <button
            type="submit"
            className="rounded-full border border-zinc-200 px-4 py-2 text-sm font-medium text-zinc-700 transition hover:border-zinc-900 hover:text-zinc-900 dark:border-zinc-700 dark:text-zinc-200 dark:hover:border-white dark:hover:text-white"
          >
            Se déconnecter
          </button>
        </form>
      }
    >
      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Chantiers</p>
          <p className="mt-2 text-3xl font-semibold">{sites?.length ?? 0}</p>
          <p className="text-sm text-zinc-500">sites suivis</p>
        </div>
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Tâches</p>
          <p className="mt-2 text-3xl font-semibold">{pendingTasks}</p>
          <p className="text-sm text-zinc-500">en attente ({doneTasks} terminées)</p>
        </div>
        <div className="rounded-2xl border border-zinc-100 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Progression</p>
          <p className="mt-2 text-3xl font-semibold">
            {totalTasks ? Math.round((doneTasks / totalTasks) * 100) : 0}%
          </p>
          <div className="mt-3 h-2 rounded-full bg-zinc-200 dark:bg-zinc-800">
            <div
              className="h-full rounded-full bg-emerald-500"
              style={{
                width: `${totalTasks ? Math.round((doneTasks / totalTasks) * 100) : 0}%`,
              }}
            />
          </div>
        </div>
      </section>

      <section className="mt-8 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-6">
          <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Ajouter un chantier
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Déclare un nouveau site avec sa deadline.
            </p>
            <div className="mt-4">
              <CreateSiteForm />
            </div>
          </div>

          <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
                Chantiers récents
              </h2>
              <span className="text-sm text-zinc-500">
                {sites?.length ?? 0} chantier(s)
              </span>
            </div>
            {error ? (
              <p className="text-sm text-rose-400">
                Impossible de charger les chantiers : {error.message}
              </p>
            ) : sites?.length ? (
              <ul className="grid gap-4 md:grid-cols-2">
                {sites.map((site) => (
                  <li
                    key={site.id}
                    className="rounded-2xl border border-zinc-100 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900"
                  >
                    <p className="text-xs uppercase tracking-[0.2em] text-zinc-500 dark:text-zinc-400">
                      Deadline
                    </p>
                    <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                      {site.deadline
                        ? new Date(site.deadline).toLocaleDateString('fr-FR')
                        : 'Non définie'}
                    </p>
                    <h3 className="mt-3 text-lg font-semibold text-zinc-900 dark:text-white">
                      {site.name}
                    </h3>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      Créé le{' '}
                      {new Date(site.created_at ?? '').toLocaleDateString('fr-FR')}
                    </p>
                    <Link
                      href={`/site/${site.id}`}
                      className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-black dark:text-white"
                    >
                      Ouvrir le chantier →
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="rounded-xl border border-dashed border-zinc-200 bg-zinc-50 p-8 text-center text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
                Aucun chantier pour le moment. Crée ton premier projet ci-dessus.
              </div>
            )}
          </div>
        </div>

        <div className="space-y-6">
          <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Prochaines deadlines
            </h2>
            <ul className="mt-4 space-y-3">
              {nextDeadlines.length ? (
                nextDeadlines.map((site) => (
                  <li
                    key={site.id}
                    className="rounded-2xl border border-zinc-200 px-4 py-3 dark:border-zinc-700"
                  >
                    <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                      {site.name}
                    </p>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      {site.deadline
                        ? new Date(site.deadline).toLocaleDateString('fr-FR', {
                            day: '2-digit',
                            month: 'short',
                            year: 'numeric',
                          })
                        : 'Deadline à planifier'}
                    </p>
                  </li>
                ))
              ) : (
                <li className="rounded-2xl border border-dashed border-zinc-200 px-4 py-3 text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
                  Pas encore de date prévue.
                </li>
              )}
            </ul>
          </div>
          <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Centralisation
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Accès direct aux interfaces terrain.
            </p>
            <div className="mt-4 grid gap-3">
              <Link
                href={sites?.[0] ? `/qr/${sites[0].id}` : '/dashboard'}
                className="rounded-2xl border border-zinc-200 px-4 py-3 text-sm font-semibold transition hover:border-zinc-900 dark:border-zinc-700 dark:hover:border-white"
              >
                🔗 QR employé
              </Link>
              <Link
                href={sites?.[0] ? `/report/${sites[0].id}` : '/dashboard'}
                className="rounded-2xl border border-zinc-200 px-4 py-3 text-sm font-semibold transition hover:border-zinc-900 dark:border-zinc-700 dark:hover:border-white"
              >
                📑 Rapports chef
              </Link>
              <Link
                href="/home"
                className="rounded-2xl border border-zinc-200 px-4 py-3 text-sm font-semibold transition hover:border-zinc-900 dark:border-zinc-700 dark:hover:border-white"
              >
                🏠 Accueil global
              </Link>
            </div>
          </div>
        </div>
      </section>
    </AppShell>
  );
}

