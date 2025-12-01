'use client';

import { useFormState } from 'react-dom';
import { useFormStatus } from 'react-dom';
import { resetPasswordAction, type AuthState } from '../actions';

const initialState: AuthState = {};

function SubmitButton() {
  const { pending } = useFormStatus();

  return (
    <button
      type="submit"
      className="w-full rounded-md bg-black py-2 text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-70"
      disabled={pending}
    >
      {pending ? 'Réinitialisation...' : 'Réinitialiser le mot de passe'}
    </button>
  );
}

export function ResetPasswordForm() {
  const [state, formAction] = useFormState(resetPasswordAction, initialState);

  return (
    <form action={formAction} className="space-y-4">
      <div className="space-y-2">
        <label
          htmlFor="password"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Nouveau mot de passe
        </label>
        <input
          id="password"
          name="password"
          type="password"
          placeholder="••••••••"
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          required
          minLength={6}
        />
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          Minimum 6 caractères
        </p>
      </div>
      <div className="space-y-2">
        <label
          htmlFor="confirmPassword"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Confirmer le mot de passe
        </label>
        <input
          id="confirmPassword"
          name="confirmPassword"
          type="password"
          placeholder="••••••••"
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          required
          minLength={6}
        />
      </div>
      {state?.error ? (
        <p className="text-sm text-rose-400">{state.error}</p>
      ) : null}
      {state?.success ? (
        <p className="text-sm text-emerald-400">{state.success}</p>
      ) : null}
      <SubmitButton />
    </form>
  );
}

