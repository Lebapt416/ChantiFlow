'use client';

import { useActionState } from 'react';
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
  const [state, formAction] = useActionState(resetPasswordRequestAction, initialState);

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
        <div className="space-y-2">
          <p className="text-sm text-rose-400">{state.error}</p>
          {state.error.includes('configuration') || state.error.includes('indisponible') ? (
            <div className="rounded-md bg-amber-50 p-3 text-xs text-amber-700 dark:bg-amber-900/20 dark:text-amber-400">
              <p className="font-medium mb-1">Vérifications à faire :</p>
              <ul className="list-disc list-inside space-y-1">
                <li>Vérifiez que l&apos;email est correct</li>
                <li>Vérifiez vos spams/courriers indésirables</li>
                <li>Assurez-vous que le service d&apos;email est configuré dans Supabase</li>
                <li>Vérifiez que l&apos;URL de redirection est autorisée dans Supabase Dashboard</li>
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}
      {state?.success ? (
        <div className="space-y-2">
          <p className="text-sm text-emerald-400">{state.success}</p>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            Le lien expire dans 1 heure. Si vous ne recevez pas l&apos;email, vérifiez vos spams.
          </p>
        </div>
      ) : null}
      <SubmitButton />
    </form>
  );
}

