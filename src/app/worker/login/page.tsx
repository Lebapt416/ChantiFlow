import { redirect } from 'next/navigation';
import { WorkerLoginForm } from './worker-login-form';
import { readWorkerSession } from '@/lib/worker-session';
import { loginWorkerWithToken } from './actions';

export const dynamic = 'force-dynamic';

export const metadata = {
  title: 'Connexion Employé | ChantiFlow',
};

type Props = {
  searchParams?: {
    token?: string;
    siteId?: string;
  };
};

export default async function WorkerLoginPage({ searchParams }: Props) {
  const tokenParam = searchParams?.token;
  let tokenError: string | null = null;

  const existingSession = await readWorkerSession();
  if (existingSession?.workerId) {
    redirect('/worker/dashboard');
  }

  if (tokenParam && typeof tokenParam === 'string') {
    const result = await loginWorkerWithToken(tokenParam);
    if (result.success) {
      redirect('/worker/dashboard');
    }
    tokenError = result.error ?? 'Impossible de vous connecter avec ce lien.';
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-50 to-zinc-100 dark:from-zinc-900 dark:to-zinc-950">
      <div className="flex min-h-screen items-center justify-center p-4">
        <div className="w-full max-w-md">
          <div className="rounded-3xl border border-zinc-200 bg-white p-8 shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
            <div className="mb-6 text-center">
              <h1 className="text-2xl font-bold text-zinc-900 dark:text-white">Espace Employé</h1>
              <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
                Scannez votre QR personnel ou saisissez votre code d&apos;accès.
              </p>
            </div>
            <WorkerLoginForm tokenError={tokenError} />
          </div>
        </div>
      </div>
    </div>
  );
}
