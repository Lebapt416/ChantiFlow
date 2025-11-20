'use client';

import { useTransition } from 'react';
import { signOutAction } from '../actions';

export function SignOutButton() {
  const [isPending, startTransition] = useTransition();

  function handleSignOut() {
    startTransition(async () => {
      await signOutAction();
    });
  }

  return (
    <button
      type="button"
      onClick={handleSignOut}
      disabled={isPending}
      className="rounded-full border border-rose-200 bg-rose-50 px-6 py-2 text-sm font-semibold text-rose-700 transition hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-rose-900/60 dark:bg-rose-900/20 dark:text-rose-300 dark:hover:bg-rose-900/30"
    >
      {isPending ? 'Déconnexion...' : 'Se déconnecter'}
    </button>
  );
}

