'use client';

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { useFormStatus } from 'react-dom';
import { LogIn, Loader2, Mail } from 'lucide-react';
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

export function WorkerLoginForm() {
  const [isPending, startTransition] = useTransition();
  const [state, setState] = useState<LoginState | null>(null);
  const router = useRouter();

  async function handleSubmit(formData: FormData) {
    startTransition(async () => {
      const result = await workerLoginAction({}, formData);
      setState(result);
      if (result.success && result.workerId && result.siteId) {
        // Rediriger vers l'espace worker
        router.push(`/worker/${result.siteId}`);
      }
    });
  }

  return (
    <form action={handleSubmit} className="space-y-4">
      <div className="space-y-2">
        <label
          htmlFor="email"
          className="block text-sm font-medium text-zinc-700 dark:text-zinc-300"
        >
          Email
        </label>
        <div className="relative">
          <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-400" />
          <input
            id="email"
            name="email"
            type="email"
            placeholder="votre@email.com"
            required
            className="w-full rounded-md border border-zinc-300 bg-white pl-10 pr-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-800 dark:text-white"
          />
        </div>
      </div>

      <div className="space-y-2">
        <label
          htmlFor="access_code"
          className="block text-sm font-medium text-zinc-700 dark:text-zinc-300"
        >
          Code d'accès
        </label>
        <input
          id="access_code"
          name="access_code"
          type="text"
          placeholder="XXXXXXXX"
          required
          maxLength={8}
          className="w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-center text-lg font-mono font-bold tracking-widest shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-800 dark:text-white"
          style={{ letterSpacing: '0.5em' }}
        />
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          Le code vous a été envoyé par email lors de l'assignation de votre tâche.
        </p>
      </div>

      {state?.error && (
        <div className="rounded-md bg-rose-50 p-3 text-sm text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">
          {state.error}
        </div>
      )}

      <SubmitButton isPending={isPending} />
    </form>
  );
}

