'use client';

import { useState, useTransition } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { LogIn, Loader2 } from 'lucide-react';
import { workerLoginAction } from './actions';

type LoginState = {
  error?: string;
  success?: boolean;
};

function SubmitButton({ isPending }: { isPending: boolean }) {
  return (
    <button
      type="submit"
      disabled={isPending}
      className="w-full rounded-md bg-emerald-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50 flex items-center justify-center gap-2"
    >
      {isPending ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          Connexion...
        </>
      ) : (
        <>
          <LogIn className="h-4 w-4" />
          Se connecter
        </>
      )}
    </button>
  );
}

export function WorkerLoginForm({ tokenError }: { tokenError?: string | null }) {
  const [isPending, startTransition] = useTransition();
  const [state, setState] = useState<LoginState | null>(null);
  const router = useRouter();
  const searchParams = useSearchParams();
  const siteIdFromUrl = searchParams.get('siteId');

  async function handleSubmit(formData: FormData) {
    // Ajouter le siteId depuis l'URL si présent
    if (siteIdFromUrl) {
      formData.append('siteId', siteIdFromUrl);
    }

    startTransition(async () => {
      const result = await workerLoginAction({}, formData);
      setState(result);
      if (result.success && result.workerId) {
        router.push('/worker/dashboard');
        router.refresh();
      }
    });
  }

  return (
    <form action={handleSubmit} className="space-y-4">
      <div className="space-y-2">
        <label
          htmlFor="access_code"
          className="block text-sm font-medium text-zinc-700 dark:text-zinc-300"
        >
          Code d&apos;accès
        </label>
        <input
          id="access_code"
          name="access_code"
          type="text"
          placeholder="1234ABCD"
          required
          maxLength={8}
          pattern="[0-9]{4}[A-Z]{4}"
          className="w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-center text-lg font-mono font-bold tracking-widest shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-800 dark:text-white uppercase"
          style={{ letterSpacing: '0.3em' }}
          autoFocus
          onChange={(e) => {
            // Forcer le format: 4 chiffres + 4 lettres
            const value = e.target.value.toUpperCase().replace(/[^0-9A-Z]/g, '');
            // Séparer les chiffres et les lettres
            const digits = value.match(/[0-9]/g)?.join('') || '';
            const letters = value.match(/[A-Z]/g)?.join('') || '';
            // Limiter à 4 chiffres puis 4 lettres
            const formatted = (digits.slice(0, 4) + letters.slice(0, 4)).slice(0, 8);
            e.target.value = formatted;
          }}
        />
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          Format: 4 chiffres + 4 lettres (ex: 1234ABCD). Le code vous a été envoyé par email lors de l&apos;assignation de votre tâche.
        </p>
      </div>

      {(tokenError || state?.error) && (
        <div className="rounded-md bg-rose-50 p-3 text-sm text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">
          {tokenError || state?.error}
        </div>
      )}

      <SubmitButton isPending={isPending} />
    </form>
  );
}

