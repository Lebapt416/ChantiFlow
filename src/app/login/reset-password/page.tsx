import Image from 'next/image';
import Link from 'next/link';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { ResetPasswordForm } from './reset-password-form';

export const metadata = {
  title: 'Réinitialiser le mot de passe | ChantiFlow',
};

export default async function ResetPasswordPage() {
  // Vérifier que l'utilisateur a un token de réinitialisation valide
  const supabase = await createSupabaseServerClient({ allowCookieSetter: true });
  const { data: { user }, error } = await supabase.auth.getUser();

  // Si pas d'utilisateur ou erreur, le token n'est pas valide
  const isValidToken = !error && user !== null;

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
            <h1 className="text-2xl font-semibold text-zinc-900 dark:text-white">
              Nouveau mot de passe
            </h1>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Choisissez un nouveau mot de passe sécurisé.
            </p>
          </div>
        </div>
        {!isValidToken ? (
          <div className="space-y-4">
            <div className="rounded-md bg-rose-50 p-3 text-sm text-rose-700 dark:bg-rose-900/20 dark:text-rose-400">
              Lien de réinitialisation invalide ou expiré. Veuillez demander un nouveau lien.
            </div>
            <Link
              href="/login/forgot-password"
              className="block w-full rounded-md bg-black py-2 text-center text-white transition hover:bg-zinc-800"
            >
              Demander un nouveau lien
            </Link>
          </div>
        ) : (
          <ResetPasswordForm />
        )}
        <div className="mt-4 text-center">
          <Link
            href="/login"
            className="text-sm text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-300"
          >
            ← Retour à la connexion
          </Link>
        </div>
      </div>
    </div>
  );
}

