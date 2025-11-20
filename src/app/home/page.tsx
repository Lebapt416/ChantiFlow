import Link from 'next/link';
import { redirect } from 'next/navigation';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import {
  LayoutDashboard,
  FolderKanban,
  ListChecks,
  UsersRound,
  FileText,
  QrCode,
} from 'lucide-react';

export const metadata = {
  title: 'Accueil | ChantiFlow',
};

export default async function HomePage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: sites } = await supabase
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

  const progress = totalTasks > 0 ? Math.round((doneTasks / totalTasks) * 100) : 0;
  const nextDeadlines = (sites ?? [])
    .filter((site) => site.deadline)
    .sort((a, b) => (a.deadline ?? '').localeCompare(b.deadline ?? ''))
    .slice(0, 3);

  const pageCards = [
    {
      title: 'Dashboard',
      description: 'Crée des chantiers et suis les deadlines.',
      href: '/dashboard',
      icon: LayoutDashboard,
    },
    {
      title: 'Tâches',
      description: 'Vue globale des tâches en attente et terminées.',
      href: '/tasks',
      icon: ListChecks,
    },
    {
      title: 'Équipe',
      description: 'Tous les intervenants et leurs rôles.',
      href: '/team',
      icon: UsersRound,
    },
    {
      title: 'Rapports',
      description: 'Photos et remontées envoyées via QR.',
      href: '/reports',
      icon: FileText,
    },
    {
      title: 'QR codes',
      description: 'Accès employé pour chaque chantier.',
      href: '/qr',
      icon: QrCode,
    },
    {
      title: 'Dernier chantier',
      description: 'Ouvre directement la fiche la plus récente.',
      href: sites?.[0] ? `/site/${sites[0].id}` : '/dashboard',
      icon: FolderKanban,
    },
  ];

  return (
    <AppShell
      heading="Accueil"
      subheading="Vue synthétique du chantier et accès rapide à toutes les pages."
      userEmail={user.email}
      primarySite={sites?.[0] ?? null}
      actions={
        <Link
          href="/dashboard"
          className="rounded-full bg-black px-4 py-2 text-sm font-medium text-white dark:bg-white dark:text-black"
        >
          + Nouveau chantier
        </Link>
      }
    >
      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Chantiers</p>
          <p className="mt-2 text-3xl font-semibold">{sites?.length ?? 0}</p>
          <p className="text-sm text-zinc-500">actifs dans ton espace</p>
        </div>
        <div className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Progression</p>
          <p className="mt-2 text-3xl font-semibold">{progress}%</p>
          <p className="text-sm text-zinc-500">
            {doneTasks}/{totalTasks} tâches terminées
          </p>
          <div className="mt-4 h-2 rounded-full bg-zinc-200 dark:bg-zinc-800">
            <div
              className="h-full rounded-full bg-emerald-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
        <div className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500">Prochain jalon</p>
          {nextDeadlines.length ? (
            <>
              <p className="mt-2 text-lg font-semibold">
                {nextDeadlines[0].name}
              </p>
              <p className="text-sm text-zinc-500">
                Deadline{' '}
                {nextDeadlines[0].deadline
                  ? new Date(nextDeadlines[0].deadline).toLocaleDateString('fr-FR')
                  : '--'}
              </p>
            </>
          ) : (
            <p className="mt-2 text-sm text-zinc-500">Aucun chantier planifié.</p>
          )}
        </div>
      </section>

      <section className="mt-10 grid gap-6 lg:grid-cols-2">
        <div className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
            Actions rapides
          </h2>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Accède immédiatement aux principales interfaces.
          </p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <Link
              href="/dashboard"
              className="rounded-2xl border border-zinc-200 px-4 py-3 text-sm font-semibold transition hover:border-zinc-900 dark:border-zinc-700 dark:hover:border-white"
            >
              ➕ Créer un chantier
            </Link>
            <Link
              href={sites?.[0] ? `/site/${sites[0].id}` : '/dashboard'}
              className="rounded-2xl border border-zinc-200 px-4 py-3 text-sm font-semibold transition hover:border-zinc-900 dark:border-zinc-700 dark:hover:border-white"
            >
              🚧 Ouvrir la dernière fiche
            </Link>
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
          </div>
        </div>

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
                      : 'Deadline à définir'}
                  </p>
                </li>
              ))
            ) : (
              <li className="rounded-2xl border border-dashed border-zinc-200 px-4 py-3 text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
                Aucune deadline renseignée.
              </li>
            )}
          </ul>
        </div>
      </section>

      <section className="mt-10 rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
          Pages disponibles
        </h2>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Tout ChantiFlow en un coup d’œil.
        </p>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          {pageCards.map((page) => (
            <Link
              key={page.href}
              href={page.href}
              className="flex items-start gap-3 rounded-2xl border border-zinc-200 p-5 transition hover:border-zinc-900 dark:border-zinc-700 dark:hover:border-white"
            >
              <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-zinc-900/5 text-zinc-700 dark:bg-white/10 dark:text-white">
                <page.icon size={22} />
              </span>
              <div>
                <p className="text-base font-semibold text-zinc-900 dark:text-white">
                  {page.title}
                </p>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  {page.description}
                </p>
              </div>
            </Link>
          ))}
        </div>
      </section>
    </AppShell>
  );
}

