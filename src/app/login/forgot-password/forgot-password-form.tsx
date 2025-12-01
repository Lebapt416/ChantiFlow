'use client';

import { useFormState } from 'react-dom';
import { useFormStatus } from 'react-dom';
import { resetPasswordRequestAction, type AuthState } from '../actions';

const initialState: AuthState = {};

function SubmitButton() {
  const { pending } = useFormStatus();

  return (
    <button
      type="submit"
      className="w-full rounded-md bg-black py-2 text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-70"
      disabled={pending}
    >
      {pending ? 'Envoi...' : 'Envoyer le lien'}
    </button>
  );
}

export function ForgotPasswordForm() {
  const [state, formAction] = useFormState(resetPasswordRequestAction, initialState);

  return (
    <form action={formAction} className="space-y-4">
      <div className="space-y-2">
        <label
          htmlFor="email"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Email
        </label>
        <input
          id="email"
          name="email"
          type="email"
          placeholder="chef@chantiflow.com"
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          required
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

