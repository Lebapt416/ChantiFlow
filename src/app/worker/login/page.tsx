import { redirect } from 'next/navigation';
import { cookies } from 'next/headers';
import { WorkerLoginForm } from './worker-login-form';

export const dynamic = 'force-dynamic';

export const metadata = {
  title: 'Connexion Employé | ChantiFlow',
};

type Props = {
  searchParams: Promise<{ siteId?: string }>;
};

export default async function WorkerLoginPage({ searchParams }: Props) {
  const { siteId } = await searchParams;
  
  // Vérifier si l'utilisateur est déjà connecté
  const cookieStore = await cookies();
  const workerSession = cookieStore.get('worker_session');
  
  if (workerSession?.value) {
    try {
      const session = JSON.parse(workerSession.value);
      if (session.workerId && session.siteId) {
        // Si un siteId est fourni dans l'URL, vérifier qu'il correspond
        if (siteId && session.siteId === siteId) {
          redirect(`/worker/${siteId}`);
        } else if (!siteId) {
          // Pas de siteId dans l'URL, utiliser celui de la session
          redirect(`/worker/${session.siteId}`);
        }
      }
    } catch (error) {
      // Session invalide, continuer vers le formulaire
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-50 to-zinc-100 dark:from-zinc-900 dark:to-zinc-950">
      <div className="flex min-h-screen items-center justify-center p-4">
        <div className="w-full max-w-md">
          <div className="rounded-3xl border border-zinc-200 bg-white p-8 shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
            <div className="mb-6 text-center">
              <h1 className="text-2xl font-bold text-zinc-900 dark:text-white">
                Espace Employé
              </h1>
              <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
                Connectez-vous avec votre code d&apos;accès unique
              </p>
            </div>
            <WorkerLoginForm />
          </div>
        </div>
      </div>
    </div>
  );
}

