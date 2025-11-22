'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ReactNode, useMemo } from 'react';
import {
  Home,
  LayoutDashboard,
  ListChecks,
  UsersRound,
  QrCode,
  FileText,
  FolderKanban,
  Sparkles,
  User,
  Calendar,
} from 'lucide-react';

type NavItem = {
  href: string;
  label: string;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
};

type PrimarySite = {
  id: string;
  name: string;
};

type AppShellProps = {
  heading: string;
  subheading?: string;
  userEmail?: string | null;
  children: ReactNode;
  primarySite?: PrimarySite | null;
  actions?: ReactNode;
};

const baseNavItems: NavItem[] = [
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/home', label: 'Accueil', icon: Home },
  { href: '/sites', label: 'Chantiers', icon: FolderKanban },
  { href: '/planning', label: 'Planning IA', icon: Calendar },
  { href: '/tasks', label: 'Tâches', icon: ListChecks },
  { href: '/team', label: 'Équipe', icon: UsersRound },
  { href: '/reports', label: 'Rapports', icon: FileText },
  { href: '/qr', label: 'QR codes', icon: QrCode },
  { href: '/account', label: 'Mon compte', icon: User },
];

export function AppShell({
  heading,
  subheading,
  userEmail,
  children,
  primarySite,
  actions,
}: AppShellProps) {
  const pathname = usePathname();

  const navItems = useMemo(() => {
    // Détecter si on est sur une page de chantier (/site/[id]/...)
    const siteMatch = pathname.match(/^\/site\/([^/]+)/);
    const siteId = siteMatch ? siteMatch[1] : null;

    if (siteId) {
      // Navigation contextuelle au chantier sélectionné
      return [
        { href: `/site/${siteId}/dashboard`, label: 'Dashboard', icon: LayoutDashboard },
        { href: '/home', label: 'Accueil', icon: Home },
        { href: `/site/${siteId}/planning`, label: 'Planning IA', icon: Calendar },
        { href: `/site/${siteId}/tasks`, label: 'Tâches', icon: ListChecks },
        { href: '/team', label: 'Équipe', icon: UsersRound },
        { href: `/site/${siteId}/reports`, label: 'Rapports', icon: FileText },
        { href: `/site/${siteId}/qr`, label: 'QR code', icon: QrCode },
        { href: '/account', label: 'Mon compte', icon: User },
      ];
    }

    // Navigation limitée quand aucun chantier n'est sélectionné
    return [
      { href: '/home', label: 'Accueil', icon: Home },
      { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
      { href: '/qr', label: 'QR codes', icon: QrCode },
      { href: '/team', label: 'Équipe générale', icon: UsersRound },
      { href: '/account', label: 'Mon compte', icon: User },
    ];
  }, [pathname]);

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-white">
      <div className="lg:flex">
        <aside className="fixed inset-y-0 left-0 z-20 hidden w-64 flex-col border-r border-zinc-200 bg-white/80 px-5 py-8 shadow-lg shadow-black/5 backdrop-blur dark:border-zinc-800 dark:bg-zinc-900/80 lg:flex">
          <div className="mb-10">
            <p className="text-xs uppercase tracking-[0.4em] text-zinc-500 dark:text-zinc-400">
              ChantiFlow
            </p>
            <p className="mt-2 text-2xl font-semibold">Pilotage</p>
          </div>
          <nav className="flex flex-1 flex-col gap-2">
            {navItems.map((item) => {
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`inline-flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition ${
                    active
                      ? 'bg-zinc-900 text-white shadow-lg shadow-black/20 dark:bg-white dark:text-black'
                      : 'text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800'
                  }`}
                >
                  <span
                    className={`flex h-9 w-9 items-center justify-center rounded-xl ${
                      active
                        ? 'bg-white/20 text-white dark:bg-black/10 dark:text-black'
                        : 'bg-zinc-900/5 text-zinc-500 dark:bg-white/5 dark:text-zinc-200'
                    }`}
                  >
                    <item.icon size={18} strokeWidth={2.2} />
                  </span>
                  {item.label}
                </Link>
              );
            })}
          </nav>
          <div className="mt-8 rounded-2xl border border-zinc-200 p-4 text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
            {primarySite ? (
              <>
                <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                  Site actif
                </p>
                <p>{primarySite.name}</p>
                <p className="mt-2 text-[11px] uppercase tracking-wide text-zinc-500">
                  QR + Rapports dispo à gauche
                </p>
              </>
            ) : (
              <p>Aucun chantier sélectionné pour le moment.</p>
            )}
          </div>
        </aside>
        <div className="min-h-screen flex-1 lg:ml-64">
          <header className="sticky top-0 z-10 border-b border-zinc-200 bg-white/80 px-4 py-4 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/80">
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.4em] text-zinc-500 dark:text-zinc-400">
                  ChantiFlow
                </p>
                <h1 className="text-2xl font-semibold">{heading}</h1>
                {subheading ? (
                  <p className="text-sm text-zinc-500 dark:text-zinc-400">{subheading}</p>
                ) : null}
              </div>
              <div className="flex items-center gap-3">
                {actions}
                {userEmail ? (
                  <div className="rounded-full border border-zinc-200 px-4 py-2 text-sm font-medium text-zinc-700 dark:border-zinc-700 dark:text-zinc-200">
                    {userEmail}
                  </div>
                ) : null}
              </div>
            </div>
            <div className="mt-4 flex gap-2 lg:hidden">
              {navItems.map((item) => {
                const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex-1 rounded-full border px-3 py-2 text-center text-xs font-semibold transition ${
                      active
                        ? 'border-zinc-900 bg-zinc-900 text-white dark:border-white dark:bg-white dark:text-black'
                        : 'border-zinc-200 text-zinc-600 dark:border-zinc-700 dark:text-zinc-300'
                    }`}
                  >
                    <div className="flex items-center justify-center gap-1">
                      <item.icon size={14} />
                      {item.label}
                    </div>
                  </Link>
                );
              })}
            </div>
          </header>
          <main className="px-4 py-8 lg:px-10">{children}</main>
        </div>
      </div>
    </div>
  );
}

