'use client';

import { useState } from 'react';
import { CreditCard, Loader2, Lock } from 'lucide-react';

type OptimalPaymentButtonProps = {
  planName?: string;
  price?: string;
  priceLabel?: string;
  onClick?: () => Promise<void> | void;
  ctaLabel?: string;
  disabled?: boolean;
  loadingLabel?: string;
  showPlanName?: boolean;
};

/**
 * Bouton de paiement premium avec signaux de confiance intégrés.
 * Peut être relié à Stripe (Checkout, Payment Link, etc.) via la prop onClick.
 */
export function OptimalPaymentButton({
  planName = 'Pro',
  price = '79',
  priceLabel,
  onClick,
  ctaLabel = "Passer à l'offre",
  disabled = false,
  loadingLabel = 'Redirection sécurisée...',
  showPlanName = true,
}: OptimalPaymentButtonProps) {
  const [isLoading, setIsLoading] = useState(false);

  const handleClick = async () => {
    if (isLoading || disabled) return;
    setIsLoading(true);
    try {
      await onClick?.();
    } finally {
      // On laisse l'état loading s'afficher (redirection vers Stripe / checkout)
    }
  };

  return (
    <div className="flex flex-col items-center gap-2">
      <button
        type="button"
        onClick={handleClick}
        disabled={isLoading || disabled}
        className="group relative w-full overflow-hidden rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 p-4 text-white shadow-lg transition-all duration-300 hover:scale-[1.02] hover:shadow-blue-500/25 hover:from-blue-500 hover:to-indigo-500 disabled:cursor-not-allowed disabled:opacity-80 min-h-[72px]"
      >
        <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/20 to-transparent transition-transform duration-1000 group-hover:translate-x-full" />

        <div className="relative flex flex-wrap items-center justify-center gap-3 text-center text-lg font-semibold sm:flex-nowrap">
          {isLoading ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              {loadingLabel}
            </>
          ) : (
            <>
              <span className="flex-1 min-w-[120px]">
                {ctaLabel} {showPlanName ? planName : ''}
              </span>
              <span className="flex items-center justify-center rounded-md bg-white/20 px-2 py-0.5 text-sm">
                {priceLabel ?? `${price}€ / mois`}
              </span>
              <CreditCard className="ml-1 h-5 w-5" />
            </>
          )}
        </div>
      </button>

      <div className="flex items-center gap-1.5 text-xs text-muted-foreground opacity-80">
        <Lock className="h-3 w-3" />
        <span>
          Paiement 100% sécurisé via <strong>Stripe</strong>
        </span>
      </div>
    </div>
  );
}

