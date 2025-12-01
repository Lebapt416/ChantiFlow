'use client';

import { useState } from 'react';
import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import Link from 'next/link';
import { signInAction, signUpAction, type AuthState } from './actions';

const initialState: AuthState = {};

function SubmitButton({ isSignUp }: { isSignUp: boolean }) {
  const { pending } = useFormStatus();

  return (
    <button
      type="submit"
      className="w-full rounded-md bg-black py-2 text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-70"
      disabled={pending}
    >
      {pending ? (isSignUp ? 'Création...' : 'Connexion...') : (isSignUp ? 'Créer un compte' : 'Se connecter')}
    </button>
  );
}

export function AuthForm() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [signInState, signInFormAction] = useActionState(signInAction, initialState);
  const [signUpState, signUpFormAction] = useActionState(signUpAction, initialState);

  const currentState = isSignUp ? signUpState : signInState;
  const currentAction = isSignUp ? signUpFormAction : signInFormAction;

  return (
    <div className="space-y-4">
      {/* Onglets */}
      <div className="flex rounded-lg bg-zinc-100 p-1 dark:bg-zinc-800">
        <button
          type="button"
          onClick={() => setIsSignUp(false)}
          className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            !isSignUp
              ? 'bg-white text-zinc-900 shadow-sm dark:bg-zinc-700 dark:text-white'
              : 'text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white'
          }`}
        >
          Connexion
        </button>
        <button
          type="button"
          onClick={() => setIsSignUp(true)}
          className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            isSignUp
              ? 'bg-white text-zinc-900 shadow-sm dark:bg-zinc-700 dark:text-white'
              : 'text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white'
          }`}
        >
          Inscription
        </button>
      </div>

      {/* Formulaire */}
      <form action={currentAction} className="space-y-4">
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
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label
              htmlFor="password"
              className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
            >
              Mot de passe
            </label>
            {!isSignUp && (
              <Link
                href="/login/forgot-password"
                className="text-xs text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-300"
              >
                Mot de passe oublié ?
              </Link>
            )}
          </div>
          <input
            id="password"
            name="password"
            type="password"
            placeholder="••••••••"
            className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
            required
            minLength={6}
          />
          {isSignUp && (
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Minimum 6 caractères
            </p>
          )}
        </div>
        {isSignUp && (
          <div className="space-y-2">
            <label
              htmlFor="name"
              className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
            >
              Nom (optionnel)
            </label>
            <input
              id="name"
              name="name"
              type="text"
              placeholder="Votre nom"
              className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
            />
          </div>
        )}
        {currentState?.error ? (
          <p className="text-sm text-rose-400">{currentState.error}</p>
        ) : null}
        {currentState?.success ? (
          <p className="text-sm text-emerald-400">{currentState.success}</p>
        ) : null}
        <SubmitButton isSignUp={isSignUp} />
      </form>
    </div>
  );
}

