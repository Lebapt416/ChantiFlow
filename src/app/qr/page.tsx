import Link from 'next/link';
import { redirect } from 'next/navigation';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { QrCode } from 'lucide-react';

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
    .select('id, name, deadline, created_at')
    .eq('created_by', user.id)
    .order('created_at', { ascending: false });

  return (
    <AppShell
      heading="QR codes"
      subheading="Gérez les QR codes d'accès pour vos chantiers."
      userEmail={user.email}
      primarySite={sites?.[0] ?? null}
    >
      <section className="rounded-3xl border border-zinc-100 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              QR codes disponibles
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Cliquez sur un QR code pour le voir, le tester et l'imprimer.
            </p>
          </div>
        </div>
        {sites?.length ? (
          <div className="space-y-2">
            {sites.map((site) => (
              <div
                key={site.id}
                className="flex items-center justify-between gap-4 rounded-lg border border-zinc-200 bg-white p-4 transition hover:border-zinc-300 hover:shadow-sm dark:border-zinc-700 dark:bg-zinc-900 dark:hover:border-zinc-600"
              >
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <QrCode className="h-5 w-5 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-zinc-900 dark:text-white truncate">
                      {site.name}
                    </p>
                    {site.deadline && (
                      <p className="text-xs text-zinc-500 dark:text-zinc-400 truncate">
                        Deadline: {new Date(site.deadline).toLocaleDateString('fr-FR')}
                      </p>
                    )}
                  </div>
                </div>
                <Link
                  href={`/qr/${site.id}/view`}
                  className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-emerald-700 active:scale-95 shadow-sm hover:shadow-md whitespace-nowrap flex-shrink-0"
                >
                  <QrCode className="h-4 w-4" />
                  <span>Voir QR code</span>
                </Link>
              </div>
            ))}
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

