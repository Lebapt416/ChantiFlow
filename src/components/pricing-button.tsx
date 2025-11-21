'use client';

import { useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { changePlanAction } from '@/app/account/actions';
import { isAdminUser } from '@/lib/stripe';

type Props = {
  plan: 'basic' | 'plus' | 'pro';
  isAuthenticated: boolean;
  className?: string;
  children?: React.ReactNode;
  userEmail?: string | null;
};

export function PricingButton({ plan, isAuthenticated, className, children, userEmail }: Props) {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();

  async function handleClick() {
    if (!isAuthenticated) {
      router.push('/login');
      return;
    }

    console.log('🔄 Clic sur plan (landing):', plan);

    // Plan Basic : changement gratuit
    if (plan === 'basic') {
      startTransition(async () => {
        try {
          const formData = new FormData();
          formData.append('plan', plan);
          const result = await changePlanAction({}, formData);
          if (result.success) {
            router.push('/account');
            router.refresh();
          }
        } catch (error) {
          console.error('Error changing plan:', error);
        }
      });
      return;
    }

    // Vérifier si c'est l'admin
    const isAdmin = userEmail ? isAdminUser(userEmail) : false;
    
    // Plans payants : admin = gratuit, autres = Stripe
    if (isAdmin) {
      startTransition(async () => {
        try {
          const formData = new FormData();
          formData.append('plan', plan);
          const result = await changePlanAction({}, formData);
          if (result.success) {
            router.push('/account');
            router.refresh();
          }
        } catch (error) {
          console.error('Error changing plan:', error);
        }
      });
      return;
    }

    // Rediriger vers Stripe checkout pour les autres utilisateurs
    console.log('💳 Redirection vers Stripe pour plan (landing):', plan);
    try {
      const response = await fetch('/api/stripe/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan }),
      });

      console.log('📡 Réponse API (landing):', response.status);

      const data = await response.json();
      console.log('📦 Données reçues (landing):', data);

      if (data.url) {
        console.log('✅ Redirection vers (landing):', data.url);
        window.location.href = data.url;
      } else if (data.error) {
        console.error('❌ Erreur checkout (landing):', data.error);
        alert(`Erreur: ${data.error}`);
      } else {
        console.error('❌ Pas d\'URL ni d\'erreur dans la réponse (landing)');
        alert('Erreur: Réponse inattendue du serveur.');
      }
    } catch (error) {
      console.error('❌ Exception (landing):', error);
      alert('Erreur lors de la redirection vers le paiement.');
    }
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={isPending}
      className={className}
    >
      {isPending ? 'Chargement...' : children || 'Choisir ce plan'}
    </button>
  );
}

