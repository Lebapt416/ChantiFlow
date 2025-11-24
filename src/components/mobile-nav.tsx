'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useMemo } from 'react';
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

type NavItem = {
  href: string;
  label: string;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number; className?: string }>;
};

export function MobileNav() {
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
    <nav className="fixed bottom-4 left-4 right-4 z-50 md:hidden">
      <div className="flex items-center justify-around bg-white/80 dark:bg-zinc-900/80 backdrop-blur-xl border border-white/20 dark:border-zinc-800 rounded-full shadow-2xl shadow-black/5 px-2 py-3 overflow-x-auto scrollbar-hide">
        {navItems.map((item) => {
          const isActive =
            pathname === item.href || pathname.startsWith(`${item.href}/`);

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`relative flex flex-col items-center justify-center gap-1 px-2 py-2 rounded-full transition-all duration-300 flex-shrink-0 ${
                isActive
                  ? 'text-emerald-600 dark:text-emerald-500'
                  : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
              }`}
            >
              {/* Fond actif avec glow premium */}
              {isActive && (
                <>
                  <span className="absolute inset-0 bg-emerald-50 dark:bg-emerald-950/40 rounded-full blur-md opacity-70" />
                  <span className="absolute inset-0 bg-emerald-100/50 dark:bg-emerald-900/30 rounded-full blur-sm" />
                </>
              )}
              
              {/* Icône */}
              <span
                className={`relative z-10 transition-all duration-300 ${
                  isActive
                    ? 'scale-110 drop-shadow-lg'
                    : 'scale-100 active:scale-95'
                }`}
              >
                <item.icon
                  size={22}
                  strokeWidth={isActive ? 2.5 : 2}
                  className={
                    isActive
                      ? 'drop-shadow-sm filter brightness-110'
                      : 'transition-opacity duration-200'
                  }
                />
              </span>

              {/* Label */}
              <span
                className={`relative z-10 text-[10px] font-semibold transition-all duration-300 ${
                  isActive
                    ? 'text-emerald-600 dark:text-emerald-500 font-bold'
                    : 'text-zinc-500 dark:text-zinc-500'
                }`}
              >
                {item.label}
              </span>

              {/* Indicateur actif (point en bas avec glow) */}
              {isActive && (
                <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-emerald-600 dark:bg-emerald-500 shadow-lg shadow-emerald-500/50" />
              )}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}

