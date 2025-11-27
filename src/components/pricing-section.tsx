'use client';

import { Check } from 'lucide-react';
import { PricingButton } from './pricing-button';

const plans = [
  {
    name: 'Basic',
    price: 'Gratuit',
    description: 'Parfait pour commencer',
    features: [
      '1 chantier actif',
      'Planification IA basique',
      'QR codes pour les employés',
      'Upload de photos et rapports',
      'Support par email',
    ],
    cta: 'Commencer gratuitement',
    popular: false,
  },
  {
    name: 'Plus',
    price: '29€',
    period: '/mois',
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
    cta: 'Choisir Plus',
    popular: true,
  },
  {
    name: 'Pro',
    price: '79€',
    period: '/mois',
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
    cta: 'Choisir Pro',
    popular: false,
  },
];

type Props = {
  isAuthenticated?: boolean;
  userEmail?: string | null;
};

export function PricingSection({ isAuthenticated = false, userEmail = null }: Props) {
  return (
    <section id="pricing" className="mx-auto max-w-7xl px-6 py-20">
      <div className="mx-auto max-w-2xl text-center">
        <h2 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
          Choisissez votre formule
        </h2>
        <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
          Des tarifs transparents pour tous les besoins
        </p>
      </div>

      <div className="mx-auto mt-16 grid max-w-5xl grid-cols-1 gap-8 lg:grid-cols-3">
        {plans.map((plan) => (
          <div
            key={plan.name}
            className={`relative flex h-full flex-col rounded-3xl border p-8 shadow-lg ${
              plan.popular
                ? 'border-emerald-500 bg-emerald-50 dark:border-emerald-400 dark:bg-emerald-900/10'
                : 'border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900'
            }`}
          >
            {plan.popular && (
              <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                <span className="rounded-full bg-emerald-500 px-4 py-1 text-xs font-semibold text-white dark:bg-emerald-400 dark:text-zinc-900">
                  Populaire
                </span>
              </div>
            )}

            <div className="text-center">
              <h3 className="text-2xl font-bold text-zinc-900 dark:text-white">{plan.name}</h3>
              <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
                {plan.description}
              </p>
              <div className="mt-6 flex items-baseline justify-center gap-1">
                <span className="text-4xl font-bold tracking-tight text-zinc-900 dark:text-white">
                  {plan.price}
                </span>
                {plan.period && (
                  <span className="text-lg text-zinc-600 dark:text-zinc-400">{plan.period}</span>
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
                  <span className="text-sm text-zinc-700 dark:text-zinc-300">{feature}</span>
                </li>
              ))}
            </ul>

            <div className="mt-8">
              <PricingButton
                plan={plan.name.toLowerCase() as 'basic' | 'plus' | 'pro'}
                isAuthenticated={isAuthenticated}
                userEmail={userEmail}
                ctaLabel={plan.cta}
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
        ))}
      </div>
    </section>
  );
}

