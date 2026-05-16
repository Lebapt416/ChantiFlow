'use client';

import dynamic from 'next/dynamic';
import { HeroSection } from './hero-section-optimized';

// Lazy load tous les composants non essentiels au FCP
const FeaturesSection = dynamic(() => import('./features-section').then(mod => ({ default: mod.FeaturesSection })), {
  ssr: false,
  loading: () => <div className="h-96" />, // Placeholder pour éviter layout shift
});

const FaqSection = dynamic(() => import('./faq-section').then(mod => ({ default: mod.FaqSection })), {
  ssr: false,
  loading: () => <div className="h-64" />,
});

const Footer = dynamic(() => import('./footer').then(mod => ({ default: mod.Footer })), {
  ssr: false,
});

const PricingSection = dynamic(() => import('@/components/pricing-section').then(mod => ({ default: mod.PricingSection })), {
  ssr: false,
  loading: () => <div className="h-96" />,
});

type LandingPageClientProps = {
  faqEntries: Array<{ question: string; answer: string }>;
};

export function LandingPageClient({ faqEntries }: LandingPageClientProps) {
  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950">
      <div className="sticky top-0 z-50 flex h-9 items-center justify-center bg-amber-100 px-4 text-center text-xs font-medium text-amber-800 dark:bg-amber-900 dark:text-amber-200">
        Version préliminaire — ChantiFlow est en phase de bêta privée. Inscription bientôt disponible.
      </div>
      <HeroSection />

      {/* Sections statiques restantes */}
      {/* ... autres sections ... */}

      {/* Features Section */}
      <FeaturesSection />

      {/* Pricing Section */}
      <PricingSection isAuthenticated={false} userEmail={null} />

      {/* FAQ Section */}
      <FaqSection entries={faqEntries} />

      {/* Footer */}
      <Footer />
    </div>
  );
}

