import Link from 'next/link';
import { redirect } from 'next/navigation';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { CopyButton } from '@/components/copy-button';
import { SiteQrCard } from '@/app/site/[id]/qr-card';

export const metadata = {
  title: 'QR Codes | ChantiFlow',
};

export default async function QrHubPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: sites } = await supabase
    .from('sites')
    .select('id, name, deadline')
    .eq('created_by', user.id)
    .order('created_at', { ascending: true });

  const appUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? 'http://localhost:3000';

  return (
    <AppShell
      heading="QR codes"
      subheading="Centralise les accès employés pour tous tes chantiers."
      userEmail={user.email}
      primarySite={sites?.[0] ?? null}
    >
      <section className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-4 flex flex-col gap-2">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
            Liens disponibles
          </h2>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Partage le QR imprimé ou l’URL directe pour un accès sans authentification.
          </p>
        </div>
        {sites?.length ? (
          <div className="grid gap-6 md:grid-cols-2">
            {sites.map((site) => {
              const url = `${appUrl.replace(/\/$/, '')}/qr/${site.id}`;
              return (
                <div key={site.id} className="space-y-4">
                  <SiteQrCard siteName={site.name} targetUrl={url} />
                  <div className="flex gap-2">
                    <Link
                      href={`/qr/${site.id}`}
                      className="flex-1 rounded-full border border-zinc-200 bg-white px-4 py-2 text-center text-xs font-semibold text-zinc-700 transition hover:border-zinc-900 hover:bg-zinc-50 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-100 dark:hover:border-white"
                    >
                      Ouvrir la page employé
                    </Link>
                    <CopyButton value={url} />
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Aucun chantier encore créé.
          </p>
        )}
      </section>
    </AppShell>
  );
}

