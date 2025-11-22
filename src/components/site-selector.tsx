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
      // Détecter la route actuelle pour rediriger vers la même page du nouveau chantier
      const path = window.location.pathname;
      
      if (path.startsWith('/site/') && path.includes('/')) {
        // Extraire la partie après /site/[id]/
        const parts = path.split('/');
        const siteIndex = parts.indexOf('site');
        if (siteIndex !== -1 && parts[siteIndex + 1]) {
          const subPath = parts.slice(siteIndex + 2).join('/');
          if (subPath) {
            router.push(`/site/${siteId}/${subPath}`);
          } else {
            router.push(`/site/${siteId}/dashboard`);
          }
        } else {
          router.push(`/site/${siteId}/dashboard`);
        }
      } else if (path.startsWith('/planning')) {
        router.push(`/planning?site=${siteId}`);
      } else {
        router.push(`/site/${siteId}/dashboard`);
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

