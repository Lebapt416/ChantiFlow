'use client';

import { useState } from 'react';
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
      <div className="flex border border-rule mb-6">
        <button
          type="button"
          onClick={() => {
            setIsSignUp(false);
            setError(null);
            setSuccess(null);
          }}
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
          onClick={() => {
            setIsSignUp(true);
            setError(null);
            setSuccess(null);
          }}
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
      <form onSubmit={handleSubmit} className="space-y-4">
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
            value={email}
            onChange={(e) => setEmail(e.target.value)}
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
            value={password}
            onChange={(e) => setPassword(e.target.value)}
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
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Votre nom"
              className="w-full px-4 py-3 bg-paper border border-rule text-ink font-sans focus:outline-none focus:border-orange transition-colors"
            />
          </div>
        )}
        {error && (
          <p className="text-sm text-danger mt-2">{error}</p>
        )}
        {success && (
          <div className="space-y-1">
            <p className="text-sm text-green">{success}</p>
          </div>
        )}
        <button
          type="submit"
          className="w-full bg-orange text-paper px-6 py-3 font-medium text-[15px] transition-colors hover:bg-orange-dark disabled:opacity-60 disabled:cursor-not-allowed"
          disabled={loading}
        >
          {loading
            ? isSignUp ? 'Création...' : 'Connexion...'
            : isSignUp ? 'Créer un compte' : 'Se connecter'}
        </button>
      </form>
    </div>
  );
}
