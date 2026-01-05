'use client';

import { useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { changePlanAction } from '@/app/account/actions';
import { isAdminUser } from '@/lib/stripe';
import { OptimalPaymentButton } from '@/components/payments/optimal-payment-button';

type Props = {
  plan: 'basic' | 'plus' | 'pro';
  isAuthenticated: boolean;
  userEmail?: string | null;
  ctaLabel: string;
  isAnnual?: boolean;
};

export function PricingButton({ plan, isAuthenticated, userEmail, ctaLabel, isAnnual = false }: Props) {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const planNames: Record<Props['plan'], string> = {
    basic: 'Basic',
    plus: 'Plus',
    pro: 'Pro',
  };
  const planPricesMonthly: Record<Props['plan'], string> = {
    basic: '0',
    plus: '29',
    pro: '79',
  };
  const planPricesAnnual: Record<Props['plan'], string> = {
    basic: '0',
    plus: '278',
    pro: '758',
  };
  
  const currentPrice = isAnnual ? planPricesAnnual[plan] : planPricesMonthly[plan];
  const priceLabel = plan === 'basic' 
    ? 'Gratuit' 
    : isAnnual 
      ? `${currentPrice}€ / an`
      : `${currentPrice}€ / mois`;

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
      const STRIPE_CHECKOUT_LINKS_MONTHLY = {
        plus: 'https://buy.stripe.com/6oUfZh8dFeSC3UbcG32VG00',
        pro: 'https://buy.stripe.com/9B6dR951t6m6aizfSf2VG01',
      };
      
      const STRIPE_CHECKOUT_LINKS_ANNUAL = {
        plus: 'https://buy.stripe.com/aFa3cv79BaCmbmD49x2VG04',
        pro: 'https://buy.stripe.com/cNibJ1alN9yi62j6hF2VG05',
      };
      
      const checkoutLinks = isAnnual ? STRIPE_CHECKOUT_LINKS_ANNUAL : STRIPE_CHECKOUT_LINKS_MONTHLY;
      const checkoutLink = checkoutLinks[plan as 'plus' | 'pro'];
      if (checkoutLink) {
        console.log('✅ Redirection directe vers Stripe (non connecté):', checkoutLink, isAnnual ? '(annuel)' : '(mensuel)');
        window.location.href = checkoutLink;
        return;
      }
    }

    // Si connecté, utiliser l'API pour pré-remplir l'email
    try {
      const response = await fetch('/api/stripe/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan, isAnnual }),
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
    <OptimalPaymentButton
      planName={planNames[plan]}
      price={currentPrice}
      priceLabel={priceLabel}
      onClick={handleClick}
      disabled={isPending}
      ctaLabel={ctaLabel}
      loadingLabel={plan === 'basic' ? 'Activation...' : 'Redirection sécurisée...'}
      showPlanName={plan !== 'basic'}
    />
  );
}

