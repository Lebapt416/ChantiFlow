import Image from 'next/image';
import Link from 'next/link';
import { ForgotPasswordForm } from './forgot-password-form';

export const metadata = {
  title: 'Mot de passe oublié | ChantiFlow',
};

export default function ForgotPasswordPage() {
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
              Mot de passe oublié
            </h1>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Entrez votre email pour recevoir un lien de réinitialisation.
            </p>
          </div>
        </div>
        <ForgotPasswordForm />
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

