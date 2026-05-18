'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Calendar, FileText, Home, CheckSquare, QrCode } from 'lucide-react';

const items = [
  { href: '/worker/dashboard', label: 'Accueil', icon: Home },
  { href: '/worker/tasks', label: 'Tâches', icon: CheckSquare },
  { href: '/worker/planning', label: 'Planning', icon: Calendar },
  { href: '/worker/reports', label: 'Rapports', icon: FileText },
  { href: '/worker/scanner', label: 'Scanner', icon: QrCode },
];

export function WorkerNav() {
  const pathname = usePathname();

  return (
    <nav className="fixed bottom-4 left-4 right-4 z-50 flex items-center justify-between rounded-full border border-rule-soft bg-paper px-4 py-3  shadow-black/10  dark:border-zinc-800/60  md:hidden">
      {items.map((item) => {
        const isActive = pathname === item.href;
        const Icon = item.icon;
        return (
          <Link
            key={item.href}
            href={item.href}
            className={`flex flex-1 flex-col items-center gap-1 rounded-full px-2 py-1 text-xs font-semibold transition ${
              isActive
                ? 'text-orange dark:text-green'
                : 'text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white'
            }`}
          >
            <span
              className={`flex h-10 w-10 items-center justify-center rounded-full ${
                isActive
                  ? 'bg-paper-2 shadow-inner shadow-orange dark:bg-paper-2'
                  : 'bg-paper dark:bg-zinc-900/40'
              }`}
            >
              <Icon className="h-5 w-5" />
            </span>
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}

