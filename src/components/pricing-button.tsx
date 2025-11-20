'use client';

import { useTransition } from 'react';
import { useRouter } from 'next/navigation';

type Props = {
  plan: 'basic' | 'plus' | 'pro';
  isAuthenticated: boolean;
  className?: string;
  children?: React.ReactNode;
};

export function PricingButton({ plan, isAuthenticated, className, children }: Props) {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();

  async function handleClick() {
    if (!isAuthenticated) {
      router.push('/login');
      return;
    }

    startTransition(async () => {
      try {
        const formData = new FormData();
        formData.append('plan', plan);
        
        const response = await fetch('/api/account/change-plan', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          router.push('/account');
          router.refresh();
        }
      } catch (error) {
        console.error('Error changing plan:', error);
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

