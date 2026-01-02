'use client';

import { useState } from 'react';
import { LogOut } from 'lucide-react';
import { signOutAction } from '@/app/actions';

export function SignOutButton() {
  const [isLoading, setIsLoading] = useState(false);

  const handleSignOut = async () => {
    setIsLoading(true);
    try {
      await signOutAction();
    } catch (error) {
      console.error('Erreur lors de la déconnexion:', error);
      setIsLoading(false);
    }
  };

  return (
    <button
      onClick={handleSignOut}
      disabled={isLoading}
      className="flex items-center gap-2 rounded-lg bg-rose-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-700 disabled:cursor-not-allowed disabled:opacity-50"
    >
      <LogOut size={16} />
      {isLoading ? 'Déconnexion...' : 'Se déconnecter'}
    </button>
  );
}

