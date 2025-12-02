import Stripe from 'stripe';

// Initialiser Stripe seulement si la clé API est disponible
export const stripe = process.env.STRIPE_SECRET_KEY
  ? new Stripe(process.env.STRIPE_SECRET_KEY, {
      apiVersion: '2025-11-17.clover',
    })
  : null;

// Configuration des Price IDs Stripe (à configurer dans les variables d'environnement)
// Ces Price IDs doivent être créés dans le dashboard Stripe > Products
export const STRIPE_PRICE_IDS = {
  plus: {
    monthly: process.env.STRIPE_PRICE_ID_PLUS_MONTHLY || '',
    annual: process.env.STRIPE_PRICE_ID_PLUS_ANNUAL || '',
  },
  pro: {
    monthly: process.env.STRIPE_PRICE_ID_PRO_MONTHLY || '',
    annual: process.env.STRIPE_PRICE_ID_PRO_ANNUAL || '',
  },
} as const;

// Liens Stripe Checkout directs (fallback si Price IDs non configurés)
export const STRIPE_CHECKOUT_LINKS = {
  plus: {
    monthly: 'https://buy.stripe.com/6oUfZh8dFeSC3UbcG32VG00',
    annual: 'https://buy.stripe.com/aFa3cv79BaCmbmD49x2VG04',
  },
  pro: {
    monthly: 'https://buy.stripe.com/9B6dR951t6m6aizfSf2VG01',
    annual: 'https://buy.stripe.com/cNibJ1alN9yi62j6hF2VG05',
  },
} as const;

// Vérifier si les Price IDs sont configurés
export function hasStripePriceIds(): boolean {
  return !!(
    STRIPE_PRICE_IDS.plus.monthly &&
    STRIPE_PRICE_IDS.plus.annual &&
    STRIPE_PRICE_IDS.pro.monthly &&
    STRIPE_PRICE_IDS.pro.annual
  );
}

export function isAdminUser(email: string | null | undefined): boolean {
  return email === 'admin@gmail.com';
}

