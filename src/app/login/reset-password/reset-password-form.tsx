'use client';

import { useState, FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';

export function ResetPasswordForm() {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);

    // Validation
    if (!password || !confirmPassword) {
      setError('Les deux champs de mot de passe sont requis.');
      return;
    }

    if (password.length < 6) {
      setError('Le mot de passe doit contenir au moins 6 caractères.');
      return;
    }

    if (password !== confirmPassword) {
      setError('Les mots de passe ne correspondent pas.');
      return;
    }

    setIsLoading(true);

    try {
      const supabase = createSupabaseBrowserClient();

      // Vérifier que l'utilisateur a une session valide
      const { data: { user }, error: userError } = await supabase.auth.getUser();

      if (userError || !user) {
        console.error('[ResetPasswordForm] Erreur utilisateur:', userError);
        setError('Lien de réinitialisation invalide ou expiré. Veuillez demander un nouveau lien.');
        setIsLoading(false);
        return;
      }

      console.log('[ResetPasswordForm] Mise à jour du mot de passe pour:', user.email);

      // Mettre à jour le mot de passe
      const { error: updateError } = await supabase.auth.updateUser({
        password,
      });

      if (updateError) {
        console.error('[ResetPasswordForm] Erreur updateUser:', updateError);
        setError(updateError.message || 'Erreur lors de la mise à jour du mot de passe.');
        setIsLoading(false);
        return;
      }

      console.log('[ResetPasswordForm] Mot de passe mis à jour avec succès');

      // Déconnecter l'utilisateur pour qu'il se reconnecte avec le nouveau mot de passe
      await supabase.auth.signOut();

      // Rediriger vers la page de connexion avec un message de succès
      router.push('/login?reset=success');
    } catch (err) {
      console.error('[ResetPasswordForm] Erreur inattendue:', err);
      setError('Erreur lors de la réinitialisation. Veuillez réessayer.');
      setIsLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
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
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          required
          minLength={6}
          disabled={isLoading}
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
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          required
          minLength={6}
          disabled={isLoading}
        />
      </div>
      {error ? (
        <p className="text-sm text-rose-400">{error}</p>
      ) : null}
      <button
        type="submit"
        className="w-full rounded-md bg-black py-2 text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-70"
        disabled={isLoading}
      >
        {isLoading ? 'Réinitialisation...' : 'Réinitialiser le mot de passe'}
      </button>
    </form>
  );
}

