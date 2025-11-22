import { notFound, redirect } from 'next/navigation';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { QRCodeViewer } from '@/app/qr/[siteId]/view/qr-code-viewer';
import { TestWorkerButton } from '@/app/qr/[siteId]/view/test-worker-button';
import { PrintQrButton } from '@/app/qr/[siteId]/view/print-qr-button';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';

type Params = {
  params: Promise<{
    id: string;
  }>;
};

function isValidUuid(value: string | undefined) {
  return Boolean(value && /^[0-9a-fA-F-]{36}$/.test(value));
}

export default async function SiteQrPage({ params }: Params) {
  const { id } = await params;

  if (!isValidUuid(id)) {
    notFound();
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: site, error: siteError } = await supabase
    .from('sites')
    .select('id, name, deadline, created_by')
    .eq('id', id)
    .single();

  if (siteError || !site || site.created_by !== user.id) {
    notFound();
  }

  const appUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? 'http://localhost:3000';
  const qrUrl = `${appUrl.replace(/\/$/, '')}/qr/${site.id}`;

  return (
    <AppShell
      heading={`QR Code - ${site.name}`}
      subheading="Affichez, testez et imprimez le QR code de ce chantier"
      userEmail={user.email}
      primarySite={{ id: site.id, name: site.name }}
    >
      <div className="space-y-6">
        {/* Bouton retour */}
        <Link
          href={`/site/${site.id}/dashboard`}
          className="inline-flex items-center gap-2 text-sm text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 transition no-print"
        >
          <ArrowLeft className="h-4 w-4" />
          Retour au dashboard
        </Link>

        {/* Carte principale */}
        <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div className="mb-6 no-print">
            <h2 className="text-xl font-bold text-zinc-900 dark:text-white mb-2">
              {site.name}
            </h2>
            {site.deadline && (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Deadline: {new Date(site.deadline).toLocaleDateString('fr-FR')}
              </p>
            )}
          </div>

          {/* QR Code */}
          <div className="mb-6 print-qr-only">
            <QRCodeViewer url={qrUrl} siteName={site.name} />
          </div>

          {/* URL */}
          <div className="mb-6 rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50 no-print">
            <p className="text-xs font-semibold text-zinc-500 dark:text-zinc-400 mb-2">
              URL du QR code
            </p>
            <p className="text-sm text-zinc-900 dark:text-white break-all font-mono">
              {qrUrl}
            </p>
          </div>

          {/* Boutons d'action */}
          <div className="grid gap-3 sm:grid-cols-2 no-print">
            <TestWorkerButton siteId={site.id} siteName={site.name} />
            <PrintQrButton />
          </div>
        </div>
      </div>
    </AppShell>
  );
}

