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
  User,
  Calendar,
} from 'lucide-react';

type PrimarySite = {
  id: string;
  name: string;
};

type AppShellProps = {
  heading?: string;
  subheading?: string;
  userEmail?: string | null;
  children: ReactNode;
  primarySite?: PrimarySite | null;
  actions?: ReactNode;
  /** Si true : pas de sticky header, pas de padding sur le main (la page gère son propre layout) */
  noHeader?: boolean;
};

export function AppShell({
  heading,
  subheading,
  userEmail,
  children,
  actions,
  noHeader = false,
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
      ];
    }

    // Navigation limitée quand aucun chantier n'est sélectionné
    return [
      { href: '/home', label: 'Accueil', icon: Home },
      { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
      { href: '/qr', label: 'QR codes', icon: QrCode },
      { href: '/team', label: 'Équipe générale', icon: UsersRound },
    ];
  }, [pathname]);

  return (
    <div className="min-h-screen bg-paper text-ink">
      <div className="lg:flex">
        {/* Sidebar */}
        <aside className="fixed inset-y-0 left-0 z-20 hidden w-56 flex-col border-r border-rule bg-ink lg:flex">
          {/* Logo */}
          <div className="px-5 py-5 border-b border-rule">
            <Link href="/home" className="flex items-center gap-3">
              <div
                className="w-8 h-8 bg-paper text-ink flex items-center justify-center font-mono text-base font-medium flex-shrink-0"
                style={{ transform: 'rotate(-3deg)' }}
              >
                C
              </div>
              <span
                className="font-serif text-lg font-semibold text-paper"
                style={{ letterSpacing: '-0.02em', fontVariationSettings: '"opsz" 144, "SOFT" 50' }}
              >
                ChantiFlow
              </span>
            </Link>
          </div>

          {/* Nav items */}
          <nav className="flex-1 overflow-y-auto py-4">
            {navItems.map((item) => {
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-3 px-5 py-2.5 text-[13px] font-medium transition-colors duration-150 border-l-2 ${
                    active
                      ? 'bg-paper/10 text-paper border-orange'
                      : 'text-paper/60 border-transparent hover:text-paper hover:border-paper/30'
                  }`}
                >
                  <item.icon size={16} strokeWidth={active ? 2.5 : 1.5} />
                  {item.label}
                </Link>
              );
            })}
          </nav>

          {/* User / bottom */}
          <div className="px-5 py-4 border-t border-rule">
            {userEmail && (
              <div className="font-mono text-[10px] uppercase tracking-widest text-paper/40 truncate">
                {userEmail}
              </div>
            )}
            <Link
              href="/account"
              className="mt-1 font-mono text-[10px] uppercase tracking-widest text-paper/60 hover:text-paper transition-colors"
            >
              → Mon compte
            </Link>
            <div className="mt-3">
              <InstallPwaButton />
            </div>
          </div>
        </aside>

        {/* Main content */}
        <div className="min-h-screen flex-1 lg:ml-56">
          {!noHeader && heading && (
            <header className="sticky top-0 z-10 border-b border-rule bg-paper px-6 py-4 flex items-center justify-between">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3">ChantiFlow</p>
                <h1
                  className="font-serif text-2xl font-normal text-ink mt-0.5"
                  style={{ fontVariationSettings: '"opsz" 60, "SOFT" 30' }}
                >
                  {heading}
                </h1>
                {subheading && <p className="text-sm text-ink-2 mt-0.5">{subheading}</p>}
              </div>
              <div className="flex items-center gap-3">{actions}</div>
            </header>
          )}
          <main className={noHeader ? '' : 'px-4 py-8 lg:px-10 pb-24 md:pb-8'}>{children}</main>
        </div>
      </div>
      <MobileNav />
    </div>
  );
}
