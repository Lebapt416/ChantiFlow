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

    startTransition(async () => {
      // Plan Basic : changement gratuit
      if (plan === 'basic') {
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
        return;
      }

      // Vérifier si c'est l'admin
      const isAdmin = userEmail ? isAdminUser(userEmail) : false;
      
      // Plans payants : admin = gratuit, autres = Stripe
      if (isAdmin) {
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
        return;
      }

      // Rediriger vers Stripe checkout pour les autres utilisateurs
      try {
        const response = await fetch('/api/stripe/checkout', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ plan }),
        });

        const data = await response.json();

        if (data.url) {
          window.location.href = data.url;
        } else {
          console.error('Erreur checkout:', data.error);
        }
      } catch (error) {
        console.error('Error redirecting to checkout:', error);
      }
    });
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

