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
    <nav className="fixed bottom-0 left-0 right-0 z-50 md:hidden border-t border-rule bg-ink">
      <div className="flex items-stretch justify-around overflow-x-auto scrollbar-hide">
        {navItems.map((item) => {
          const isActive =
            pathname === item.href || pathname.startsWith(`${item.href}/`);

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`relative flex flex-col items-center justify-center gap-1 px-3 py-3 flex-shrink-0 transition-colors duration-150 ${
                isActive
                  ? 'text-paper border-t-2 border-orange'
                  : 'text-paper/50 hover:text-paper/80 border-t-2 border-transparent'
              }`}
            >
              <item.icon
                size={20}
                strokeWidth={isActive ? 2.5 : 1.5}
              />
              <span className="font-mono text-[9px] uppercase tracking-widest">
                {item.label}
              </span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
