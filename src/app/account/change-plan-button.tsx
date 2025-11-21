'use client';

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { changePlanAction, type ChangePlanState } from './actions';
import { isAdminUser } from '@/lib/stripe';

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

  return (
    <div>
      <button
        type="button"
        onClick={handleClick}
        disabled={isPending || isCurrent}
        className={`w-full rounded-lg px-3 py-2 text-xs font-semibold transition ${
          isCurrent
            ? 'bg-emerald-600 text-white dark:bg-emerald-400 dark:text-zinc-900'
            : plan === 'basic'
              ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-900'
              : 'bg-zinc-200 text-zinc-900 dark:bg-zinc-700 dark:text-white'
        } disabled:cursor-not-allowed disabled:opacity-50`}
      >
        {isPending
          ? 'Changement...'
          : isCurrent
            ? 'Plan actuel'
            : isAdmin && plan !== 'basic'
              ? 'Choisir (gratuit)'
              : plan === 'basic'
                ? 'Choisir ce plan'
                : 'Payer et activer'}
      </button>
      {state?.error && (
        <p className="mt-1 text-xs text-rose-500">{state.error}</p>
      )}
      {state?.success && (
        <p className="mt-1 text-xs text-emerald-500">Plan mis à jour !</p>
      )}
    </div>
  );
}

