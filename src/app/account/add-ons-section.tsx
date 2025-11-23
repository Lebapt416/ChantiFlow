'use client';

import { useState } from 'react';
import { Plus, Users, Building2 } from 'lucide-react';
import { type Plan, type UserAddOns } from '@/lib/plans';

type Props = {
  user: {
    id: string;
    email?: string;
  };
  currentAddOns: UserAddOns;
  plan: Plan;
};

export function AddOnsSection({ user, currentAddOns, plan }: Props) {
  const [isLoading, setIsLoading] = useState<string | null>(null);

  // Les liens Stripe seront fournis par l'utilisateur
  const STRIPE_ADDON_EXTRA_WORKERS = process.env.NEXT_PUBLIC_STRIPE_ADDON_EXTRA_WORKERS || '#';
  const STRIPE_ADDON_EXTRA_SITES = process.env.NEXT_PUBLIC_STRIPE_ADDON_EXTRA_SITES || '#';

  const handlePurchaseAddOn = async (addOnType: 'extra_workers' | 'extra_sites') => {
    setIsLoading(addOnType);
    
    try {
      const url = addOnType === 'extra_workers' 
        ? STRIPE_ADDON_EXTRA_WORKERS 
        : STRIPE_ADDON_EXTRA_SITES;
      
      if (url === '#') {
        alert('Les liens Stripe pour les add-ons ne sont pas encore configurés. Veuillez contacter le support.');
        setIsLoading(null);
        return;
      }

      // Rediriger vers Stripe Checkout
      window.location.href = url;
    } catch (error) {
      console.error('Erreur lors de l\'achat de l\'add-on:', error);
      alert('Une erreur est survenue. Veuillez réessayer.');
      setIsLoading(null);
    }
  };

  // Ne pas afficher les add-ons pour le plan pro (illimité)
  if (plan === 'pro') {
    return null;
  }

  return (
    <section className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
      <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">Extensions (Add-ons)</h2>
      <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
        Ajoutez des fonctionnalités supplémentaires à votre plan actuel.
      </p>
      
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        {/* Add-on +5 employés */}
        <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800">
          <div className="flex items-start gap-3">
            <div className="rounded-lg bg-emerald-100 p-2 dark:bg-emerald-900/30">
              <Users className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
                +5 employés
              </h3>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                10€/mois
              </p>
              <p className="mt-2 text-xs text-zinc-600 dark:text-zinc-300">
                Ajoutez 5 employés supplémentaires à votre équipe.
              </p>
              {currentAddOns.extra_workers ? (
                <p className="mt-2 text-xs font-medium text-emerald-600 dark:text-emerald-400">
                  Actif : {currentAddOns.extra_workers} add-on(s) (+{currentAddOns.extra_workers * 5} employés)
                </p>
              ) : null}
            </div>
          </div>
          <button
            onClick={() => handlePurchaseAddOn('extra_workers')}
            disabled={isLoading === 'extra_workers'}
            className="mt-4 w-full rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {isLoading === 'extra_workers' ? 'Chargement...' : 'Acheter'}
          </button>
        </div>

        {/* Add-on +2 chantiers */}
        <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800">
          <div className="flex items-start gap-3">
            <div className="rounded-lg bg-blue-100 p-2 dark:bg-blue-900/30">
              <Building2 className="h-5 w-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
                +2 chantiers
              </h3>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                5€/mois
              </p>
              <p className="mt-2 text-xs text-zinc-600 dark:text-zinc-300">
                Ajoutez 2 chantiers actifs supplémentaires.
              </p>
              {currentAddOns.extra_sites ? (
                <p className="mt-2 text-xs font-medium text-blue-600 dark:text-blue-400">
                  Actif : {currentAddOns.extra_sites} add-on(s) (+{currentAddOns.extra_sites * 2} chantiers)
                </p>
              ) : null}
            </div>
          </div>
          <button
            onClick={() => handlePurchaseAddOn('extra_sites')}
            disabled={isLoading === 'extra_sites'}
            className="mt-4 w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {isLoading === 'extra_sites' ? 'Chargement...' : 'Acheter'}
          </button>
        </div>
      </div>
    </section>
  );
}

