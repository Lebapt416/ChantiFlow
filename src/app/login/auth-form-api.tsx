'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export function AuthFormAPI() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    setLoading(true);

    try {
      if (isSignUp) {
        // Inscription - utiliser l'API route
        const response = await fetch('/api/auth/signup', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: email.trim(), password, name: name.trim() }),
        });

        const data = await response.json();

        if (!response.ok) {
          setError(data.error || 'Erreur lors de la création du compte.');
          setLoading(false);
          return;
        }

        setSuccess('Compte créé ! Vérifiez votre email pour confirmer votre compte.');
        setLoading(false);
      } else {
        // Connexion - utiliser l'API route
        const response = await fetch('/api/auth/signin', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: email.trim(), password }),
        });

        const data = await response.json();

        if (!response.ok) {
          setError(data.error || 'Erreur lors de la connexion.');
          setLoading(false);
          return;
        }

        // Rediriger vers la page appropriée
        if (data.redirectTo) {
          window.location.href = data.redirectTo;
        } else {
          router.push('/home');
        }
      }
    } catch (err) {
      console.error('Erreur:', err);
      setError('Une erreur est survenue. Veuillez réessayer.');
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Onglets */}
      <div className="flex rounded-lg bg-zinc-100 p-1 dark:bg-zinc-800">
        <button
          type="button"
          onClick={() => {
            setIsSignUp(false);
            setError(null);
            setSuccess(null);
          }}
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
          onClick={() => {
            setIsSignUp(true);
            setError(null);
            setSuccess(null);
          }}
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
      <form onSubmit={handleSubmit} className="space-y-4">
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
            value={email}
            onChange={(e) => setEmail(e.target.value)}
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
            value={password}
            onChange={(e) => setPassword(e.target.value)}
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
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Votre nom"
              className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
            />
          </div>
        )}
        {error && (
          <p className="text-sm text-rose-400">{error}</p>
        )}
        {success && (
          <div className="space-y-2">
            <p className="text-sm text-emerald-400">{success}</p>
            {!isSignUp && (
              <p className="text-xs text-zinc-500 dark:text-zinc-400">Redirection en cours...</p>
            )}
          </div>
        )}
        <button
          type="submit"
          className="w-full rounded-md bg-black py-2 text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-70"
          disabled={loading}
        >
          {loading ? (isSignUp ? 'Création...' : 'Connexion...') : (isSignUp ? 'Créer un compte' : 'Se connecter')}
        </button>
      </form>
    </div>
  );
}

