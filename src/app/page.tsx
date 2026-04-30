import { LandingPageClient } from '@/components/landing/landing-page-client';

export const metadata = {
  title: 'ChantiFlow - Gestion de chantiers simplifiée avec IA',
  description: 'Gérez vos chantiers efficacement avec planification IA, QR codes pour les employés, et suivi en temps réel.',
  keywords: [
    'logiciel gestion chantier',
    'solution BTP',
    'planification chantier ia',
    'application suivi chantier',
  ],
  alternates: {
    canonical: '/',
  },
};

// Cache la page pendant 1 heure (ISR)
export const revalidate = 3600;

const faqEntries = [
  {
    question: 'ChantiFlow convient-il aux petites équipes ?',
    answer:
      'Oui, le logiciel est utilisé par des artisans, PME et ETI. L\'interface reste simple même avec seulement quelques chantiers en cours.',
  },
  {
    question: 'Comment se fait la mise en route ?',
    answer:
      'Nous importons vos équipes et tâches existantes via un fichier Excel ou un onboarding guidé. Vous êtes opérationnel sous 48h.',
  },
  {
    question: 'Puis-je donner un accès aux sous-traitants ?',
    answer:
      'Chaque intervenant reçoit un code sécurisé via email ou QR code. Vous contrôlez les chantiers visibles et les permissions.',
  },
  {
    question: 'L\'IA respecte-t-elle les données clients ?',
    answer:
      'Oui, toutes les données sont hébergées dans l\'Union Européenne (Supabase). Les modèles IA n\'utilisent jamais vos informations pour entraîner des services externes.',
  },
];

const baseUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? 'https://www.chantiflow.com';

const softwareSchema = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareApplication',
  name: 'ChantiFlow',
  applicationCategory: 'BusinessApplication',
  operatingSystem: 'Web',
  url: `${baseUrl}/`,
  offers: {
    '@type': 'Offer',
    price: '0',
    priceCurrency: 'EUR',
    availability: 'https://schema.org/InStock',
  },
  aggregateRating: {
    '@type': 'AggregateRating',
    ratingValue: '4.9',
    reviewCount: '26',
  },
  featureList: [
    'Planning intelligent',
    'QR codes équipe',
    'Rapports photo illimités',
    'Alertes retards',
  ],
  creator: {
    '@type': 'Organization',
    name: 'ChantiFlow SAS',
    url: baseUrl,
  },
};

const faqSchema = {
  '@context': 'https://schema.org',
  '@type': 'FAQPage',
  mainEntity: faqEntries.map((entry) => ({
    '@type': 'Question',
    name: entry.question,
    acceptedAnswer: {
      '@type': 'Answer',
      text: entry.answer,
    },
  })),
};

export default async function LandingPage() {
  return (
    <>
      <script
        type="application/ld+json"
        suppressHydrationWarning
        dangerouslySetInnerHTML={{ __html: JSON.stringify(softwareSchema) }}
      />
      <script
        type="application/ld+json"
        suppressHydrationWarning
        dangerouslySetInnerHTML={{ __html: JSON.stringify(faqSchema) }}
      />
      <script
        type="application/ld+json"
        suppressHydrationWarning
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'SoftwareApplication',
            name: 'ChantiFlow',
            applicationCategory: 'BusinessApplication',
            operatingSystem: 'Web, iOS, Android',
            offers: {
              '@type': 'Offer',
              price: '0',
              priceCurrency: 'EUR',
            },
            description: 'Application de gestion de chantiers avec planification IA et suivi en temps réel.',
          }),
        }}
      />
      <LandingPageClient
        faqEntries={faqEntries}
      />
    </>
  );
}
