import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import Link from 'next/link';
import { CheckCircle2 } from 'lucide-react';

type SearchParams = {
  searchParams: Promise<{
    session_id?: string;
  }>;
};

export default async function SuccessPage({ searchParams }: SearchParams) {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const params = await searchParams;
  const sessionId = params?.session_id;

  return (
    <AppShell
      heading="Paiement réussi"
      subheading="Votre abonnement a été activé"
      userEmail={user.email}
    >
      <div className="mx-auto max-w-2xl">
        <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-8 text-center dark:border-emerald-800 dark:bg-emerald-900/20">
          <CheckCircle2 className="mx-auto h-16 w-16 text-emerald-600 dark:text-emerald-400" />
          <h2 className="mt-4 text-2xl font-semibold text-zinc-900 dark:text-white">
            Paiement réussi !
          </h2>
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            Votre abonnement a été activé avec succès.
          </p>
          {sessionId && (
            <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-500">
              Session ID: {sessionId}
            </p>
          )}
          <div className="mt-6">
            <Link
              href="/account"
              className="inline-block rounded-lg bg-emerald-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-emerald-700 dark:bg-emerald-400 dark:text-zinc-900 dark:hover:bg-emerald-300"
            >
              Voir mon compte
            </Link>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

