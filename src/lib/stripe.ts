import Stripe from 'stripe';

// Initialiser Stripe seulement si la clé API est disponible
export const stripe = process.env.STRIPE_SECRET_KEY
  ? new Stripe(process.env.STRIPE_SECRET_KEY, {
      apiVersion: '2025-11-17.clover',
    })
  : null;

export const STRIPE_PRICE_IDS = {
  plus: process.env.STRIPE_PRICE_ID_PLUS || 'price_plus',
  pro: process.env.STRIPE_PRICE_ID_PRO || 'price_pro',
} as const;

export function isAdminUser(email: string | null | undefined): boolean {
  return email === 'admin@gmail.com';
}

