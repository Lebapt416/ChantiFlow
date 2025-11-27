'use client';

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { changePlanAction, type ChangePlanState } from './actions';
import { isAdminUser } from '@/lib/stripe';
import { OptimalPaymentButton } from '@/components/payments/optimal-payment-button';

type Props = {
  plan: 'basic' | 'plus' | 'pro';
  currentPlan: 'basic' | 'plus' | 'pro';
  userEmail?: string | null;
};

export function ChangePlanButton({ plan, currentPlan, userEmail }: Props) {
  const [isPending, startTransition] = useTransition();
  const [state, setState] = useState<ChangePlanState>({});
  const router = useRouter();
  const isCurrent = plan === currentPlan;
  const isAdmin = userEmail ? isAdminUser(userEmail) : false;
  const planNames: Record<Props['plan'], string> = {
    basic: 'Basic',
    plus: 'Plus',
    pro: 'Pro',
  };
  const planPrices: Record<Props['plan'], string> = {
    basic: '0',
    plus: '29',
    pro: '79',
  };

  async function handleClick() {
    if (isCurrent || isPending) return;

    console.log('🔄 Clic sur plan:', plan, 'Admin:', isAdmin);

    // Plan Basic : changement gratuit direct
    if (plan === 'basic') {
      startTransition(async () => {
        const formData = new FormData();
        formData.append('plan', plan);
        const result = await changePlanAction({}, formData);
        setState(result);
        if (result.success) {
          router.refresh();
        }
      });
      return;
    }

    // Plans payants (Plus/Pro)
    // Si admin : changement gratuit
    if (isAdmin) {
      startTransition(async () => {
        const formData = new FormData();
        formData.append('plan', plan);
        const result = await changePlanAction({}, formData);
        setState(result);
        if (result.success) {
          router.refresh();
        }
      });
      return;
    }

    // Autres utilisateurs : rediriger vers Stripe checkout
    console.log('💳 Redirection vers Stripe pour plan:', plan);
    try {
      const response = await fetch('/api/stripe/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan }),
      });

      console.log('📡 Réponse API:', response.status);

      const data = await response.json();
      console.log('📦 Données reçues:', data);

      if (data.url) {
        console.log('✅ Redirection vers:', data.url);
        window.location.href = data.url;
      } else if (data.error) {
        console.error('❌ Erreur:', data.error);
        setState({ error: data.error });
      } else {
        console.error('❌ Pas d\'URL ni d\'erreur dans la réponse');
        setState({ error: 'Réponse inattendue du serveur.' });
      }
    } catch (error) {
      console.error('❌ Exception:', error);
      setState({ error: 'Erreur lors de la redirection vers le paiement.' });
    }
  }

  if (isCurrent) {
    return (
      <div className="rounded-lg bg-emerald-600 px-3 py-2 text-center text-xs font-semibold text-white dark:bg-emerald-400 dark:text-zinc-900">
        Plan actuel
      </div>
    );
  }

  const ctaLabel =
    plan === 'basic' ? 'Choisir ce plan' : isAdmin ? 'Activer gratuitement' : "Passer à l'offre";

  return (
    <div>
      <OptimalPaymentButton
        planName={planNames[plan]}
        price={planPrices[plan]}
        priceLabel={plan === 'basic' ? 'Gratuit' : undefined}
        ctaLabel={ctaLabel}
        disabled={isPending}
        loadingLabel={plan === 'basic' || isAdmin ? 'Activation...' : 'Redirection sécurisée...'}
        showPlanName={plan !== 'basic'}
        onClick={handleClick}
      />
      {state?.error && (
        <p className="mt-1 text-xs text-rose-500">{state.error}</p>
      )}
      {state?.success && (
        <p className="mt-1 text-xs text-emerald-500">Plan mis à jour !</p>
      )}
    </div>
  );
}

