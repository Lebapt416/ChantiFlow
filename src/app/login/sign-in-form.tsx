"use client";

import { useActionState } from "react";
import { useFormStatus } from "react-dom";
import { signInAction, type AuthState } from './actions';

const initialState: AuthState = {};

function SubmitButton() {
  const { pending } = useFormStatus();

  return (
    <button
      type="submit"
      className="w-full bg-orange text-paper px-6 py-3 font-medium text-[15px] transition-colors hover:bg-orange-dark disabled:opacity-60 disabled:cursor-not-allowed"
      disabled={pending}
    >
      {pending ? 'Connexion...' : 'Se connecter'}
    </button>
  );
}

export function SignInForm() {
  const [state, formAction] = useActionState(signInAction, initialState);

  return (
    <form action={formAction} className="space-y-4">
      <div className="space-y-2">
        <label
          htmlFor="email"
          className="block font-mono text-[11px] uppercase tracking-widest text-ink-2 mb-2"
        >
          Email
        </label>
        <input
          id="email"
          name="email"
          type="email"
          placeholder="chef@chantiflow.com"
          className="w-full px-4 py-3 bg-paper border border-rule text-ink font-sans focus:outline-none focus:border-orange transition-colors"
          required
        />
      </div>
      <div className="space-y-2">
        <label
          htmlFor="password"
          className="block font-mono text-[11px] uppercase tracking-widest text-ink-2 mb-2"
        >
          Mot de passe
        </label>
        <input
          id="password"
          name="password"
          type="password"
          placeholder="••••••••"
          className="w-full px-4 py-3 bg-paper border border-rule text-ink font-sans focus:outline-none focus:border-orange transition-colors"
          required
        />
      </div>
      {state?.error ? (
        <p className="text-sm text-danger mt-2">{state.error}</p>
      ) : null}
      <SubmitButton />
    </form>
  );
}
