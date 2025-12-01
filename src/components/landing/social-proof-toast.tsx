'use client';

import { useEffect, useState } from 'react';
import { User } from 'lucide-react';

export function SocialProofToast() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Afficher le toast après 3 secondes
    const timer = setTimeout(() => {
      setIsVisible(true);
    }, 3000);

    return () => clearTimeout(timer);
  }, []);

  if (!isVisible) return null;

  return (
    <div
      className={`fixed bottom-6 left-6 z-50 transition-all duration-500 ease-out ${
        isVisible ? 'translate-y-0 opacity-100' : 'translate-y-8 opacity-0'
      }`}
    >
      <div className="flex items-center gap-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-4 py-3 shadow-lg backdrop-blur-sm">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-900/30">
          <User className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
        </div>
        <div className="flex-1">
          <p className="text-sm font-medium text-zinc-900 dark:text-white">
            Patrick (75) vient de créer un chantier Rénovation.
          </p>
        </div>
      </div>
    </div>
  );
}

