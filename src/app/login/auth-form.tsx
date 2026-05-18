'use client';

import { useState, useEffect } from 'react';
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
      className="w-full bg-orange text-paper px-6 py-3 font-medium text-[15px] transition-colors hover:bg-orange-dark disabled:opacity-60 disabled:cursor-not-allowed"
      disabled={pending}
    >
      {pending
        ? isSignUp ? 'Création...' : 'Connexion...'
        : isSignUp ? 'Créer un compte' : 'Se connecter'}
    </button>
  );
}

export function AuthForm() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [signInState, signInFormAction] = useActionState(signInAction, initialState);
  const [signUpState, signUpFormAction] = useActionState(signUpAction, initialState);

  const currentState = isSignUp ? signUpState : signInState;
  const currentAction = isSignUp ? signUpFormAction : signInFormAction;

  // Rediriger immédiatement après une connexion réussie
  useEffect(() => {
    if (currentState?.success && currentState?.redirectTo && !isSignUp) {
      // Utiliser window.location.href pour une redirection complète qui recharge les cookies
      window.location.href = currentState.redirectTo;
    }
  }, [currentState?.success, currentState?.redirectTo, isSignUp]);

  return (
    <div className="space-y-4">
      {/* Onglets */}
      <div className="flex border border-rule mb-6">
        <button
          type="button"
          onClick={() => setIsSignUp(false)}
          className={`flex-1 px-4 py-2.5 font-mono text-[11px] uppercase tracking-widest transition-colors duration-150 ${
            !isSignUp
              ? 'bg-ink text-paper'
              : 'bg-paper text-ink-2 hover:text-ink'
          }`}
        >
          Connexion
        </button>
        <button
          type="button"
          onClick={() => setIsSignUp(true)}
          className={`flex-1 px-4 py-2.5 font-mono text-[11px] uppercase tracking-widest transition-colors duration-150 border-l border-rule ${
            isSignUp
              ? 'bg-ink text-paper'
              : 'bg-paper text-ink-2 hover:text-ink'
          }`}
        >
          Inscription
        </button>
      </div>

      {/* Formulaire */}
      <form action={currentAction} className="space-y-4">
        <div>
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
        <div>
          <div className="flex items-center justify-between mb-2">
            <label
              htmlFor="password"
              className="font-mono text-[11px] uppercase tracking-widest text-ink-2"
            >
              Mot de passe
            </label>
            {!isSignUp && (
              <Link
                href="/login/forgot-password"
                className="font-mono text-[11px] uppercase tracking-widest text-ink-2 hover:text-ink transition-colors underline-offset-4"
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
            className="w-full px-4 py-3 bg-paper border border-rule text-ink font-sans focus:outline-none focus:border-orange transition-colors"
            required
            minLength={6}
          />
          {isSignUp && (
            <p className="mt-1.5 font-mono text-[10px] text-ink-3">
              Minimum 6 caractères
            </p>
          )}
        </div>
        {isSignUp && (
          <div>
            <label
              htmlFor="name"
              className="block font-mono text-[11px] uppercase tracking-widest text-ink-2 mb-2"
            >
              Nom (optionnel)
            </label>
            <input
              id="name"
              name="name"
              type="text"
              placeholder="Votre nom"
              className="w-full px-4 py-3 bg-paper border border-rule text-ink font-sans focus:outline-none focus:border-orange transition-colors"
            />
          </div>
        )}
        {currentState?.error ? (
          <p className="text-sm text-danger mt-2">{currentState.error}</p>
        ) : null}
        {currentState?.success ? (
          <div className="space-y-1">
            <p className="text-sm text-green">{currentState.success}</p>
            <p className="font-mono text-[10px] text-ink-3">Redirection en cours...</p>
          </div>
        ) : null}
        <SubmitButton isSignUp={isSignUp} />
      </form>
    </div>
  );
}
