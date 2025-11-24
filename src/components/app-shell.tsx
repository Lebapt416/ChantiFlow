'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ReactNode, useMemo } from 'react';
import { InstallPwaButton } from './install-pwa-button';
import { MobileNav } from './mobile-nav';
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
        { href: `/site/${siteId}/team`, label: 'Équipe', icon: UsersRound },
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
        <aside className="fixed inset-y-0 left-0 z-20 hidden w-16 flex-col items-center border-r border-zinc-800 bg-black/80 px-0 py-8 shadow-xl backdrop-blur dark:border-zinc-700 dark:bg-black/80 lg:flex">
          <nav className="flex flex-1 flex-col items-center gap-2 w-full">
            {navItems.map((item) => {
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`group/item relative flex items-center justify-center w-14 h-14 rounded-xl transition-all duration-200 ${
                    active
                      ? 'bg-white text-black shadow-lg shadow-white/20'
                      : 'text-white hover:text-white'
                  }`}
                  title={item.label}
                >
                  <span className={`absolute rounded-xl transition-all duration-200 ${
                    active
                      ? 'inset-0 bg-white'
                      : 'top-0 bottom-0 left-2 right-0 bg-black/50 group-hover/item:bg-black/70 group-hover/item:left-3'
                  }`}></span>
                  <span className="relative z-10">
                    <item.icon 
                      size={26} 
                      strokeWidth={active ? 3.5 : 3}
                      className={active ? '' : 'group-hover/item:scale-110 transition-transform duration-200'}
                    />
                  </span>
                  {/* Tooltip pour affichage au survol */}
                  <span className="pointer-events-none absolute left-full ml-3 z-50 whitespace-nowrap rounded-lg bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white opacity-0 shadow-xl transition-all duration-200 group-hover/item:opacity-100 group-hover/item:translate-x-0 -translate-x-1 dark:bg-zinc-100 dark:text-zinc-900">
                    {item.label}
                    <span className="absolute right-full top-1/2 -mr-1 h-2 w-2 -translate-y-1/2 rotate-45 bg-zinc-900 dark:bg-zinc-100"></span>
                  </span>
                </Link>
              );
            })}
            {/* Bouton d'installation PWA */}
            <div className="mt-auto pt-4">
              <InstallPwaButton />
            </div>
          </nav>
        </aside>
        <div className="min-h-screen flex-1 transition-all duration-300 ease-in-out lg:ml-16">
          <header className="sticky top-0 z-10 border-b border-zinc-200 bg-white/80 px-4 py-4 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/80">
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div className="flex items-center gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.4em] text-zinc-500 dark:text-zinc-400">
                    ChantiFlow
                  </p>
                  <h1 className="text-2xl font-semibold">{heading}</h1>
                  {subheading ? (
                    <p className="text-sm text-zinc-500 dark:text-zinc-400">{subheading}</p>
                  ) : null}
                </div>
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
          </header>
          <main className="px-4 py-8 lg:px-10 pb-24 md:pb-8">{children}</main>
        </div>
      </div>
      <MobileNav />
    </div>
  );
}

