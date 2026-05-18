'use client';

import dynamic from 'next/dynamic';
import { HeroSection } from './hero-section-optimized';

// Lazy load tous les composants non essentiels au FCP
const FeaturesSection = dynamic(() => import('./features-section').then(mod => ({ default: mod.FeaturesSection })), {
  ssr: false,
  loading: () => <div className="h-96" />,
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
    <div className="min-h-screen bg-paper">
      <div className="bg-ink py-2.5 text-center font-mono text-[11px] tracking-widest text-paper uppercase relative z-10">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-orange mr-2.5 align-middle" style={{animation: 'pulse 2.4s ease-in-out infinite'}} />
        Bêta privée · Inscription sur invitation · Prochaine vague — juin 2026
      </div>
      <HeroSection />

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
