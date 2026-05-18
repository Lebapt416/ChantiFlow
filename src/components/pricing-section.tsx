'use client';

import { useState } from 'react';
import { PricingButton } from './pricing-button';

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
    priceAnnual: '278€',
    description: 'Pour les équipes en croissance',
    features: [
      "Jusqu'à 5 chantiers actifs",
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
    priceAnnual: '758€',
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
    <section id="pricing" className="border-t border-rule py-20">
      <div className="mx-auto max-w-7xl px-8">
        {/* Chapter header */}
        <div className="grid grid-cols-[auto_1fr] gap-20 items-end pb-16 border-b border-rule-soft mb-16">
          <div className="font-mono text-[12px] text-ink-2 tracking-widest">
            <span className="block font-serif text-[96px] italic font-light text-ink leading-none mb-2" style={{fontVariationSettings: '"opsz" 144, "SOFT" 100'}}>03</span>
            Chapitre · Tarifs
          </div>
          <div>
            <h2 className="font-serif font-normal text-ink leading-[1.02] tracking-tight" style={{fontSize: 'clamp(36px, 4vw, 56px)', fontVariationSettings: '"opsz" 144, "SOFT" 30'}}>
              Des tarifs clairs,<br/>
              sans <em className="italic font-light text-orange">surprise.</em>
            </h2>
            <p className="text-[17px] text-ink-2 mt-5 max-w-[540px] leading-relaxed">
              Commencez gratuitement. Passez au niveau supérieur quand vous êtes prêt.
            </p>

            {/* Toggle */}
            <div className="flex items-center gap-0 mt-8 border border-rule self-start">
              <button
                onClick={() => setIsAnnual(false)}
                className={`px-5 py-2.5 font-mono text-[12px] tracking-widest uppercase transition-colors ${!isAnnual ? 'bg-ink text-paper' : 'bg-paper text-ink-2 hover:text-ink'}`}
              >
                Mensuel
              </button>
              <button
                onClick={() => setIsAnnual(true)}
                className={`px-5 py-2.5 font-mono text-[12px] tracking-widest uppercase transition-colors ${isAnnual ? 'bg-ink text-paper' : 'bg-paper text-ink-2 hover:text-ink'}`}
              >
                Annuel −20%
              </button>
            </div>
          </div>
        </div>

        {/* Plans grid */}
        <div className="grid grid-cols-3 border-t border-l border-rule-soft">
          {plans.map((plan) => {
            const currentPrice = isAnnual ? plan.priceAnnual : plan.priceMonthly;
            const currentPeriod = isAnnual ? '/an' : '/mois';
            const currentCta = isAnnual ? plan.ctaAnnual : plan.ctaMonthly;
            const monthlyEquivalent = isAnnual && plan.name !== 'Basic'
              ? `soit ${Math.round(parseFloat(plan.priceAnnual.replace('€', '')) / 12)}€/mois`
              : '';

            return (
              <div
                key={plan.name}
                className={`p-8 border-r border-b border-rule-soft flex flex-col ${
                  plan.popular
                    ? 'border-l-2 border-l-orange'
                    : ''
                }`}
              >
                {/* Plan name */}
                <div className="font-mono text-[11px] uppercase tracking-widest text-ink-2 mb-8">
                  {plan.name}
                  {plan.popular && (
                    <span className="ml-3 border border-orange text-orange px-2 py-0.5 text-[10px]">Populaire</span>
                  )}
                  {plan.showAnnualBadge && isAnnual && (
                    <span className="ml-3 border border-ink text-ink px-2 py-0.5 text-[10px]">2 MOIS OFFERTS</span>
                  )}
                </div>

                {/* Price */}
                <div>
                  <div className="font-serif text-[56px] leading-none tracking-tight text-ink">
                    {currentPrice}
                  </div>
                  {plan.name !== 'Basic' && (
                    <div className="font-mono text-[12px] text-ink-2 tracking-widest mt-1">{currentPeriod}</div>
                  )}
                  {monthlyEquivalent && (
                    <div className="font-mono text-[11px] text-orange mt-1 tracking-widest">{monthlyEquivalent}</div>
                  )}
                </div>

                {/* Description */}
                <p className="text-[15px] text-ink-2 mt-3">{plan.description}</p>

                {/* Separator */}
                <div className="border-t border-rule-soft my-6" />

                {/* Features */}
                <ul className="space-y-3 flex-1">
                  {plan.features.map((feature) => (
                    <li key={feature} className="flex items-start gap-2.5 text-[14px] text-ink-2">
                      <span className="text-ink-3 flex-shrink-0 font-mono">→</span>
                      {feature}
                    </li>
                  ))}
                </ul>

                {/* CTA */}
                <div className="mt-8">
                  <PricingButton
                    plan={plan.name.toLowerCase() as 'basic' | 'plus' | 'pro'}
                    isAuthenticated={isAuthenticated}
                    userEmail={userEmail}
                    ctaLabel={currentCta}
                    isAnnual={isAnnual}
                  />
                  <p className="mt-2 text-center font-mono text-[10px] text-ink-3 tracking-widest uppercase">
                    {isAuthenticated
                      ? plan.name === 'Basic'
                        ? 'Activer ce plan (gratuit)'
                        : 'Payer et activer'
                      : 'Connexion requise'}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
