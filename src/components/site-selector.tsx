'use client';

import { useRouter } from 'next/navigation';
import { FolderKanban } from 'lucide-react';

type Site = {
  id: string;
  name: string;
  deadline: string | null;
};

type Props = {
  sites: Site[];
  currentSiteId: string;
};

export function SiteSelector({ sites, currentSiteId }: Props) {
  const router = useRouter();

  function handleChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const siteId = e.target.value;
    if (siteId) {
      // Détecter si on est sur /planning ou /site
      const path = window.location.pathname;
      if (path.startsWith('/planning')) {
        router.push(`/planning?site=${siteId}`);
      } else {
        router.push(`/site/${siteId}`);
      }
    }
  }

  if (sites.length === 0) {
    return null;
  }

  return (
    <div className="flex items-center gap-2">
      <FolderKanban className="h-4 w-4 text-zinc-500 dark:text-zinc-400" />
      <select
        value={currentSiteId}
        onChange={handleChange}
        className="rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-900 dark:text-white"
      >
        {sites.map((site) => (
          <option key={site.id} value={site.id}>
            {site.name}
            {site.deadline ? ` (${new Date(site.deadline).toLocaleDateString('fr-FR')})` : ''}
          </option>
        ))}
      </select>
    </div>
  );
}

