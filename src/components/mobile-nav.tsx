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
      <div className="flex items-center justify-around bg-paper border border-rule-soft rounded px-2 py-3 overflow-x-auto scrollbar-hide">
        {navItems.map((item) => {
          const isActive =
            pathname === item.href || pathname.startsWith(`${item.href}/`);

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`relative flex flex-col items-center justify-center gap-1 px-2 py-2 rounded transition-all duration-300 flex-shrink-0 ${
                isActive
                  ? 'text-orange'
                  : 'text-ink-2 hover:text-ink'
              }`}
            >
              {/* Fond actif avec glow premium */}
              {isActive && (
                <span className="absolute inset-0 bg-paper-2 rounded" />
              )}
              
              {/* Icône */}
              <span
                className={`relative z-10 transition-all duration-300 ${
                  isActive
                    ? 'scale-110 drop-'
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
                    ? 'text-orange font-bold'
                    : 'text-ink-3'
                }`}
              >
                {item.label}
              </span>

              {/* Indicateur actif (point en bas avec glow) */}
              {isActive && (
                <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-orange" />
              )}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}

