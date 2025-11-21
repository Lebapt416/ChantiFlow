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
    console.log('🔄 Clic sur plan (landing):', plan, 'Authentifié:', isAuthenticated);

    // Plan Basic : nécessite une connexion
    if (plan === 'basic') {
      if (!isAuthenticated) {
        router.push('/login');
        return;
      }
      
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

    // Plans payants (Plus/Pro) : redirection directe vers Stripe
    // Même si l'utilisateur n'est pas connecté, on peut le rediriger vers Stripe
    // Le webhook mettra à jour le plan après le paiement si l'email correspond à un compte
    
    // Si connecté, vérifier si c'est l'admin
    if (isAuthenticated) {
      const isAdmin = userEmail ? isAdminUser(userEmail) : false;
      
      // Admin : changement gratuit
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
    }

    // Rediriger vers Stripe checkout (connecté ou non)
    console.log('💳 Redirection vers Stripe pour plan (landing):', plan);
    
    // Si l'utilisateur n'est pas connecté, rediriger directement vers le lien Stripe
    if (!isAuthenticated) {
      const STRIPE_CHECKOUT_LINKS = {
        plus: 'https://buy.stripe.com/6oUfZh8dFeSC3UbcG32VG00',
        pro: 'https://buy.stripe.com/9B6dR951t6m6aizfSf2VG01',
      };
      
      const checkoutLink = STRIPE_CHECKOUT_LINKS[plan as 'plus' | 'pro'];
      if (checkoutLink) {
        console.log('✅ Redirection directe vers Stripe (non connecté):', checkoutLink);
        window.location.href = checkoutLink;
        return;
      }
    }

    // Si connecté, utiliser l'API pour pré-remplir l'email
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

