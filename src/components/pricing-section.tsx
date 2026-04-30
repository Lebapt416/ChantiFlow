'use client';

import { useState } from 'react';
import { Check } from 'lucide-react';
import { PricingButton } from './pricing-button';
import { motion } from 'framer-motion';

type Plan = {
  name: 'Basic' | 'Plus' | 'Pro';
  priceMonthly: string;
  priceAnnual: string;
  description: string;
  features: string[];
  ctaMonthly: string;
  ctaAnnual: string;
  popular: boolean;
  showAnnualBadge?: boolean;
};

const plans: Plan[] = [
  {
    name: 'Basic',
    priceMonthly: 'Gratuit',
    priceAnnual: 'Gratuit',
    description: 'Parfait pour commencer',
    features: [
      '1 chantier actif',
      'Planification IA basique',
      'QR codes pour les employés',
      'Upload de photos et rapports',
      'Support par email',
    ],
    ctaMonthly: 'Commencer gratuitement',
    ctaAnnual: 'Commencer gratuitement',
    popular: false,
  },
  {
    name: 'Plus',
    priceMonthly: '29€',
    priceAnnual: '278€', // 29 * 12 * 0.8 = 278.4, arrondi à 278
    description: 'Pour les équipes en croissance',
    features: [
      'Jusqu\'à 5 chantiers actifs',
      'Planification IA avancée',
      'Classement intelligent des tâches',
      'Analytics détaillés',
      'Support prioritaire',
      'Export de rapports',
      'Météo de chantier en temps réel',
    ],
    ctaMonthly: 'Choisir Plus',
    ctaAnnual: 'Choisir le plan',
    popular: true,
  },
  {
    name: 'Pro',
    priceMonthly: '79€',
    priceAnnual: '758€', // 79 * 12 * 0.8 = 758.4, arrondi à 758
    description: 'Pour les grandes entreprises',
    features: [
      'Tout de Plus',
      'Multi-utilisateurs',
      'API personnalisée',
      'Intégrations avancées',
      'Support dédié 24/7',
      'Formation personnalisée',
      'Gestion des permissions',
      'Exports rapports PDF professionnels',
      'Assistant IA de chantier',
    ],
    ctaMonthly: 'Choisir Pro',
    ctaAnnual: 'Choisir le plan',
    popular: false,
    showAnnualBadge: true,
  },
];

type Props = {
  isAuthenticated?: boolean;
  userEmail?: string | null;
};

export function PricingSection({ isAuthenticated = false, userEmail = null }: Props) {
  const [isAnnual, setIsAnnual] = useState(false);

  return (
    <section id="pricing" className="mx-auto max-w-7xl px-6 py-20">
      <div className="mx-auto max-w-2xl text-center">
        <h2 className="text-3xl font-extrabold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
          Choisissez votre formule
        </h2>
        <p className="mt-4 text-lg text-zinc-700 dark:text-zinc-300 font-medium">
          Des tarifs transparents pour tous les besoins
        </p>

        {/* Toggle Liquid Glass */}
        <div className="mt-8 flex items-center justify-center gap-4">
          <span className={`text-sm font-medium transition-colors ${!isAnnual ? 'text-zinc-900 dark:text-white' : 'text-zinc-500 dark:text-zinc-400'}`}>
            Mensuel
          </span>
          <button
            onClick={() => setIsAnnual(!isAnnual)}
            className="relative h-12 w-24 rounded-full border border-white/20 dark:border-zinc-700/50 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-md shadow-lg transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
            aria-label="Basculer entre mensuel et annuel"
          >
            <motion.div
              className="absolute top-1 left-1 h-10 w-10 rounded-full bg-gradient-to-br from-emerald-500 to-emerald-600 shadow-lg"
              animate={{
                x: isAnnual ? 48 : 0,
              }}
              transition={{
                type: 'spring',
                stiffness: 500,
                damping: 30,
              }}
            >
              <div className="absolute inset-0 rounded-full bg-white/20 backdrop-blur-sm" />
            </motion.div>
          </button>
          <span className={`text-sm font-medium transition-colors ${isAnnual ? 'text-zinc-900 dark:text-white' : 'text-zinc-500 dark:text-zinc-400'}`}>
            Annuel <span className="text-emerald-600 dark:text-emerald-400">(-20%)</span>
          </span>
        </div>
      </div>

      <div className="mx-auto mt-16 grid max-w-7xl grid-cols-1 gap-6 md:grid-cols-3 md:gap-4 lg:gap-6">
        {plans.map((plan) => {
          const currentPrice = isAnnual ? plan.priceAnnual : plan.priceMonthly;
          const currentPeriod = isAnnual ? '/an' : '/mois';
          const currentCta = isAnnual ? plan.ctaAnnual : plan.ctaMonthly;
          const monthlyEquivalent = isAnnual && plan.name !== 'Basic' 
            ? `(${Math.round(parseFloat(plan.priceAnnual.replace('€', '')) / 12)}€/mois)` 
            : '';

          return (
            <div
              key={plan.name}
              className={`relative flex h-full flex-col rounded-3xl border border-white/20 dark:border-zinc-800/50 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md p-6 md:p-8 shadow-lg ${
                plan.popular
                  ? 'border-emerald-500/50 dark:border-emerald-400/50 bg-emerald-50/80 dark:bg-emerald-900/20'
                  : ''
              }`}
            >
              {plan.popular && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                  <span className="rounded-full bg-emerald-500 px-4 py-1 text-xs font-semibold text-white dark:bg-emerald-400 dark:text-zinc-900">
                    Populaire
                  </span>
                </div>
              )}
              {plan.showAnnualBadge && isAnnual && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2 z-10">
                  <span className="rounded-full bg-gradient-to-r from-orange-500 to-red-500 px-5 py-2 text-xs font-bold text-white shadow-lg animate-pulse">
                    🔥 2 MOIS OFFERTS
                  </span>
                </div>
              )}

              <div className="text-center">
                <h3 className="text-2xl font-extrabold text-zinc-900 dark:text-white">{plan.name}</h3>
                <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-300 font-medium">
                  {plan.description}
                </p>
                <div className="mt-6 flex flex-col items-center justify-center gap-1">
                  <div className="flex items-baseline gap-1">
                    <span className="text-4xl font-extrabold tracking-tight text-zinc-900 dark:text-white">
                      {currentPrice}
                    </span>
                    {plan.name !== 'Basic' && (
                      <span className="text-lg text-zinc-600 dark:text-zinc-400">{currentPeriod}</span>
                    )}
                  </div>
                  {monthlyEquivalent && (
                    <span className="text-sm text-emerald-600 dark:text-emerald-400 font-medium">
                      {monthlyEquivalent}
                    </span>
                  )}
                </div>
              </div>

              <ul className="mt-8 flex-1 space-y-4">
                {plan.features.map((feature) => (
                  <li key={feature} className="flex items-start gap-3">
                    <Check
                      className={`mt-0.5 h-5 w-5 flex-shrink-0 ${
                        plan.popular
                          ? 'text-emerald-600 dark:text-emerald-400'
                          : 'text-zinc-600 dark:text-zinc-400'
                      }`}
                    />
                    <span className="text-sm text-zinc-700 dark:text-zinc-300 font-medium">{feature}</span>
                  </li>
                ))}
              </ul>

              <div className="mt-8">
                <PricingButton
                  plan={plan.name.toLowerCase() as 'basic' | 'plus' | 'pro'}
                  isAuthenticated={isAuthenticated}
                  userEmail={userEmail}
                  ctaLabel={currentCta}
                  isAnnual={isAnnual}
                />
                <p className="mt-2 text-center text-xs text-zinc-500 dark:text-zinc-400">
                  {isAuthenticated
                    ? plan.name === 'Basic'
                      ? 'Cliquez pour activer ce plan (gratuit)'
                      : 'Cliquez pour payer et activer'
                    : 'Connectez-vous pour choisir un plan'}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

