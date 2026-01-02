import Image from 'next/image';
import { AuthFormAPI } from './auth-form-api';

export const metadata = {
  title: 'Connexion | ChantiFlow',
};

export default function LoginPage({
  searchParams,
}: {
  searchParams: { reset?: string };
}) {
  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 px-4 py-12 dark:bg-zinc-950">
      <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-2xl shadow-black/5 dark:bg-zinc-900">
        <div className="mb-8 space-y-4 text-center">
          <Image
            src="/vercel.svg"
            alt="Logo ChantiFlow"
            width={48}
            height={48}
            className="mx-auto dark:invert"
          />
          <div>
            <p className="text-sm uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
              Chantiflow
            </p>
            <h1 className="text-2xl font-semibold text-zinc-900 dark:text-white">
              Espace chef de chantier
            </h1>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Connecte-toi ou crée un compte pour commencer.
            </p>
          </div>
        </div>
        {searchParams?.reset === 'success' && (
          <div className="mb-4 rounded-md bg-emerald-50 p-3 text-sm text-emerald-700 dark:bg-emerald-900/20 dark:text-emerald-400">
            Mot de passe réinitialisé avec succès ! Vous pouvez maintenant vous connecter.
          </div>
        )}
        <AuthFormAPI />
      </div>
    </div>
  );
}

