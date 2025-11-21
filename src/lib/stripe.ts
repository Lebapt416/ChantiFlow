import Stripe from 'stripe';

// Initialiser Stripe seulement si la clé API est disponible
export const stripe = process.env.STRIPE_SECRET_KEY
  ? new Stripe(process.env.STRIPE_SECRET_KEY, {
      apiVersion: '2025-11-17.clover',
    })
  : null;

// Liens Stripe Checkout directs
export const STRIPE_CHECKOUT_LINKS = {
  plus: 'https://buy.stripe.com/6oUfZh8dFeSC3UbcG32VG00',
  pro: 'https://buy.stripe.com/9B6dR951t6m6aizfSf2VG01',
} as const;

export function isAdminUser(email: string | null | undefined): boolean {
  return email === 'admin@gmail.com';
}

