import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import Link from 'next/link';
import { XCircle } from 'lucide-react';

export default async function CancelPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  return (
    <AppShell
      heading="Paiement annulé"
      subheading="Vous pouvez réessayer à tout moment"
      userEmail={user.email}
    >
      <div className="mx-auto max-w-2xl">
        <div className="rounded-2xl border border-zinc-200 bg-white p-8 text-center dark:border-zinc-800 dark:bg-zinc-900">
          <XCircle className="mx-auto h-16 w-16 text-zinc-400" />
          <h2 className="mt-4 text-2xl font-semibold text-zinc-900 dark:text-white">
            Paiement annulé
          </h2>
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            Vous avez annulé le processus de paiement. Aucun montant n'a été débité.
          </p>
          <div className="mt-6 flex gap-3 justify-center">
            <Link
              href="/account"
              className="inline-block rounded-lg bg-zinc-900 px-6 py-3 text-sm font-semibold text-white transition hover:bg-zinc-800 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              Retour au compte
            </Link>
            <Link
              href="/dashboard"
              className="inline-block rounded-lg border border-zinc-200 px-6 py-3 text-sm font-semibold text-zinc-700 transition hover:border-zinc-900 hover:text-zinc-900 dark:border-zinc-700 dark:text-zinc-200 dark:hover:border-white dark:hover:text-white"
            >
              Aller au dashboard
            </Link>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

