import type { LucideIcon } from 'lucide-react';
import Link from 'next/link';
import {
  Check,
  Zap,
  Users,
  QrCode,
  Brain,
  Calendar,
  Clock,
  AlertTriangle,
  MessageSquare,
  FileText,
  Camera,
  TrendingUp,
  Shield,
  HardHat,
  Hammer,
  Building2,
  Ruler,
  LineChart,
} from 'lucide-react';
import { PricingSection } from '@/components/pricing-section';
import { createSupabaseServerClient } from '@/lib/supabase/server';

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

type SectorHighlight = {
  title: string;
  description: string;
  icon: LucideIcon;
};

const sectorHighlights: SectorHighlight[] = [
  {
    title: 'Entreprises générales du BTP',
    description:
      'Pilotez simultanément les lots maçonnerie, gros œuvre et second œuvre avec un même tableau de bord.',
    icon: Building2,
  },
  {
    title: 'Artisans & PME du bâtiment',
    description:
      'Centralisez les interventions plomberie, électricité, peinture ou rénovation sans multiplier les outils.',
    icon: Hammer,
  },
  {
    title: 'Conducteurs de travaux',
    description:
      'Suivez budgets, retards et ressources en temps réel pour chaque chantier, même en mobilité.',
    icon: HardHat,
  },
  {
    title: 'Bureaux d’études / MOE',
    description:
      'Disposez d’une traçabilité complète (rapports, photos, réserves) pour garantir la conformité de vos livrables.',
    icon: Ruler,
  },
];

const faqEntries = [
  {
    question: 'ChantiFlow convient-il aux petites équipes ?',
    answer:
      'Oui, le logiciel est utilisé par des artisans, PME et ETI. L’interface reste simple même avec seulement quelques chantiers en cours.',
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
    question: 'L’IA respecte-t-elle les données clients ?',
    answer:
      'Oui, toutes les données sont hébergées dans l’Union Européenne (Supabase). Les modèles IA n’utilisent jamais vos informations pour entraîner des services externes.',
  },
];

const baseUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? 'https://www.chantiflow.com';

const proofMetrics = [
  { value: '30%', label: 'productivité gagnée', description: 'Gràce au planning IA et aux alertes.' },
  { value: '2h', label: 'temps économisé par chef de chantier', description: 'Rapports et relances automatisés.' },
  { value: '100%', label: 'traçabilité des rapports', description: 'Photos, réserves et documents centralisés.' },
];

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
  // Vérifier si l'utilisateur est connecté
  let isAuthenticated = false;
  try {
    const supabase = await createSupabaseServerClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();
    isAuthenticated = !!user;
  } catch {
    isAuthenticated = false;
  }
  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950">
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
      {/* Header */}
      <header className="border-b border-zinc-200 dark:border-zinc-800">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-black text-white dark:bg-white dark:text-black">
                <Zap className="h-6 w-6" />
              </div>
              <span className="text-xl font-bold text-zinc-900 dark:text-white">
                ChantiFlow
              </span>
            </div>
            <nav className="flex items-center gap-6">
              <Link
                href="/login"
                className="text-sm font-medium text-zinc-600 transition hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
              >
                Connexion
              </Link>
              <Link
                href="/login"
                className="rounded-full bg-black px-5 py-2 text-sm font-semibold text-white transition hover:bg-zinc-800 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
              >
                Commencer
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="text-sm font-semibold uppercase tracking-[0.4em] text-emerald-600 dark:text-emerald-400">
            Logiciel de gestion BTP
          </h2>
          <h1 className="text-5xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-6xl">
            Logiciel de Suivi de Chantier&nbsp;: gérez vos chantiers{' '}
            <span className="text-emerald-600 dark:text-emerald-400">30% plus vite</span>{' '}
            grâce à l'IA
          </h1>
          <p className="mt-6 text-lg leading-8 text-zinc-600 dark:text-zinc-400">
            ChantiFlow automatise votre planning, centralise vos équipes, génère vos rapports, et suit l'avancement de vos chantiers en temps réel — sans Excel, sans appels inutiles.
          </p>
          <div className="mt-10 flex flex-col items-center justify-center gap-4">
            <Link
              href="/login"
              className="rounded-full bg-black px-8 py-4 text-lg font-bold text-white shadow-xl transition hover:bg-zinc-800 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
            >
              Démarrer gratuitement
            </Link>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              ✓ Compatible smartphone et QR codes de chantier
            </p>
          </div>
        </div>
      </section>

      {/* Comment ça marche */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
            Comment ça marche ?
          </h2>
          <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
            En 3 étapes simples, transformez votre gestion de chantier
          </p>
        </div>

        <div className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-8 sm:grid-cols-3 lg:max-w-none">
          <div className="rounded-2xl border border-zinc-200 bg-white p-8 text-center shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-900/30">
              <span className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">1</span>
            </div>
            <h3 className="mt-6 text-lg font-semibold text-zinc-900 dark:text-white">
              Créez votre chantier
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Définissez vos tâches, équipes et deadlines en quelques clics
            </p>
          </div>

          <div className="rounded-2xl border border-zinc-200 bg-white p-8 text-center shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/30">
              <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">2</span>
            </div>
            <h3 className="mt-6 text-lg font-semibold text-zinc-900 dark:text-white">
              Liez vos équipes
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Partagez un QR code : vos employés accèdent instantanément à leurs tâches
            </p>
          </div>

          <div className="rounded-2xl border border-zinc-200 bg-white p-8 text-center shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-purple-100 dark:bg-purple-900/30">
              <span className="text-2xl font-bold text-purple-600 dark:text-purple-400">3</span>
            </div>
            <h3 className="mt-6 text-lg font-semibold text-zinc-900 dark:text-white">
              L'IA optimise tout
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Planification automatique, suivi en temps réel, rapports générés automatiquement
            </p>
          </div>
        </div>
      </section>

      {/* Pourquoi ChantiFlow ? */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
            Pourquoi ChantiFlow ?
          </h2>
          <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
            Les problèmes du BTP, résolus par l'IA
          </p>
        </div>

        <div className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-8 sm:grid-cols-3 lg:max-w-none">
          <div className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-rose-100 dark:bg-rose-900/30">
              <Clock className="h-6 w-6 text-rose-600 dark:text-rose-400" />
            </div>
            <h3 className="mt-6 text-lg font-semibold text-zinc-900 dark:text-white">
              Retards de chantier
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              <span className="font-semibold text-rose-600 dark:text-rose-400">Problème :</span> Délais non respectés, planning désorganisé
            </p>
            <p className="mt-4 text-sm font-semibold text-emerald-600 dark:text-emerald-400">
              ✓ Solution : Planning optimisé par IA, alertes automatiques, suivi en temps réel
            </p>
          </div>

          <div className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-100 dark:bg-amber-900/30">
              <MessageSquare className="h-6 w-6 text-amber-600 dark:text-amber-400" />
            </div>
            <h3 className="mt-6 text-lg font-semibold text-zinc-900 dark:text-white">
              Mauvaise communication
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              <span className="font-semibold text-rose-600 dark:text-rose-400">Problème :</span> Appels multiples, informations perdues
            </p>
            <p className="mt-4 text-sm font-semibold text-emerald-600 dark:text-emerald-400">
              ✓ Solution : QR codes instantanés, rapports centralisés, notifications automatiques
            </p>
          </div>

          <div className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-indigo-100 dark:bg-indigo-900/30">
              <FileText className="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
            </div>
            <h3 className="mt-6 text-lg font-semibold text-zinc-900 dark:text-white">
              Perte d'informations
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              <span className="font-semibold text-rose-600 dark:text-rose-400">Problème :</span> Photos perdues, rapports éparpillés
            </p>
            <p className="mt-4 text-sm font-semibold text-emerald-600 dark:text-emerald-400">
              ✓ Solution : Stockage centralisé, photos liées aux rapports, historique complet
            </p>
          </div>
        </div>
      </section>

      {/* Secteurs & métiers ciblés */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-2xl text-center">
          <p className="text-sm font-semibold uppercase tracking-widest text-emerald-600 dark:text-emerald-400">
            Compatible avec tous les métiers du BTP
          </p>
          <h2 className="mt-4 text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
            Le logiciel pensé pour votre réalité terrain
          </h2>
          <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
            ChantiFlow couvre la planification, le suivi sécurité et la traçabilité documentaire pour les chantiers
            de construction, rénovation, voirie ou maintenance industrielle.
          </p>
        </div>
        <div className="mx-auto mt-16 grid max-w-4xl grid-cols-1 gap-6 sm:grid-cols-2">
          {sectorHighlights.map((sector) => (
            <div
              key={sector.title}
              className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm transition hover:-translate-y-1 hover:border-emerald-300 dark:border-zinc-800 dark:bg-zinc-900"
            >
              <sector.icon className="h-8 w-8 text-emerald-600 dark:text-emerald-400" />
              <h3 className="mt-4 text-xl font-semibold text-zinc-900 dark:text-white">{sector.title}</h3>
              <p className="mt-2 text-sm leading-6 text-zinc-600 dark:text-zinc-400">{sector.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
            Tout ce dont vous avez besoin
          </h2>
          <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
            Des outils puissants pour optimiser vos projets de construction
          </p>
        </div>

        {/* Image du dashboard */}
        <div className="mx-auto mt-12 max-w-5xl">
          <div className="rounded-2xl border border-zinc-200 bg-gradient-to-br from-zinc-50 to-zinc-100 p-4 shadow-xl dark:border-zinc-800 dark:from-zinc-900 dark:to-zinc-950">
            <div className="rounded-lg overflow-hidden bg-zinc-900">
              <div className="aspect-video bg-zinc-900 flex items-center justify-center relative">
                {/* Simulation du dashboard avec des éléments visuels améliorés */}
                <div className="w-full h-full p-6 space-y-4">
                  <div className="grid grid-cols-3 gap-3">
                    <div className="bg-zinc-800 rounded-lg p-3">
                      <div className="text-xs text-zinc-400 mb-1">CHANTIERS</div>
                      <div className="text-2xl font-bold text-white">2</div>
                      <div className="text-xs text-zinc-500">sites suivis</div>
                    </div>
                    <div className="bg-zinc-800 rounded-lg p-3">
                      <div className="text-xs text-zinc-400 mb-1">TÂCHES</div>
                      <div className="text-2xl font-bold text-white">4</div>
                      <div className="text-xs text-zinc-500">en attente</div>
                    </div>
                    <div className="bg-zinc-800 rounded-lg p-3">
                      <div className="text-xs text-zinc-400 mb-1">PROGRESSION</div>
                      <div className="text-2xl font-bold text-emerald-400">0%</div>
                      <div className="w-full bg-zinc-700 rounded-full h-1.5 mt-2">
                        <div className="bg-emerald-400 h-1.5 rounded-full" style={{ width: '0%' }}></div>
                      </div>
                    </div>
                  </div>
                  <div className="bg-zinc-800 rounded-lg p-4">
                    <div className="text-sm font-semibold text-white mb-2">Chantiers récents</div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-zinc-300">Test chantier 2</span>
                        <span className="text-xs text-zinc-500">27/11/2025</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-zinc-300">Test chantier 1</span>
                        <span className="text-xs text-zinc-500">23/11/2025</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 text-center mt-3">
              Interface de gestion de chantier ChantiFlow
            </p>
          </div>
        </div>

        <div className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-6 sm:grid-cols-2 lg:max-w-none lg:grid-cols-3">
          <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center gap-3">
              <Brain className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
              <h3 className="text-base font-semibold text-zinc-900 dark:text-white">
                Planning automatique
              </h3>
            </div>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center gap-3">
              <QrCode className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              <h3 className="text-base font-semibold text-zinc-900 dark:text-white">
                QR codes pour les équipes
              </h3>
            </div>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center gap-3">
              <FileText className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              <h3 className="text-base font-semibold text-zinc-900 dark:text-white">
                Rapports auto
              </h3>
            </div>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center gap-3">
              <Camera className="h-5 w-5 text-amber-600 dark:text-amber-400" />
              <h3 className="text-base font-semibold text-zinc-900 dark:text-white">
                Photos & documents
              </h3>
            </div>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center gap-3">
              <TrendingUp className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
              <h3 className="text-base font-semibold text-zinc-900 dark:text-white">
                Suivi en temps réel
              </h3>
            </div>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center gap-3">
              <AlertTriangle className="h-5 w-5 text-rose-600 dark:text-rose-400" />
              <h3 className="text-base font-semibold text-zinc-900 dark:text-white">
                Alerte retards
              </h3>
            </div>
          </div>
        </div>
      </section>

      {/* Résultats constatés */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
            Des gains mesurables sur vos chantiers
          </h2>
          <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
            Les entreprises utilisant ChantiFlow améliorent leur visibilité chantier et leur marge grâce à des
            données temps réel.
          </p>
        </div>
        <div className="mx-auto mt-12 grid max-w-4xl grid-cols-1 gap-6 sm:grid-cols-3">
          {proofMetrics.map((metric) => (
            <div key={metric.label} className="rounded-2xl border border-zinc-200 bg-white p-6 text-center dark:border-zinc-800 dark:bg-zinc-900">
              <LineChart className="mx-auto h-8 w-8 text-emerald-600 dark:text-emerald-400" />
              <p className="mt-4 text-4xl font-black text-zinc-900 dark:text-white">{metric.value}</p>
              <p className="mt-2 text-sm font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">{metric.label}</p>
              <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">{metric.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Pricing Section */}
      <PricingSection isAuthenticated={isAuthenticated} userEmail={null} />

      {/* À propos */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-3xl">
          <div className="rounded-3xl border border-zinc-200 bg-white p-12 text-center dark:border-zinc-800 dark:bg-zinc-900">
            <h2 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
              Qui sommes-nous ?
            </h2>
            <p className="mt-6 text-lg leading-8 text-zinc-600 dark:text-zinc-400">
              Nous sommes trois jeunes entrepreneurs passionnés par la modernisation du BTP. 
              Face aux défis quotidiens des chantiers — retards, communication difficile, perte d'informations — 
              nous avons créé ChantiFlow pour apporter une solution simple, efficace et assistée par l'IA.
            </p>
            <p className="mt-4 text-base text-zinc-600 dark:text-zinc-400">
              Notre mission : rendre la gestion de chantier accessible à tous, sans complexité inutile.
            </p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-2xl rounded-3xl bg-zinc-900 px-6 py-16 text-center dark:bg-zinc-800">
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Prêt à optimiser vos chantiers ?
          </h2>
          <p className="mt-6 text-lg leading-8 text-zinc-300">
            Rejoignez les équipes qui font confiance à ChantiFlow pour gérer leurs projets de
            construction.
          </p>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <Link
              href="/login"
              className="rounded-full bg-white px-8 py-4 text-lg font-bold text-zinc-900 shadow-xl transition hover:bg-zinc-100"
            >
              Commencer gratuitement
            </Link>
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-3xl text-center">
          <p className="text-sm font-semibold uppercase tracking-widest text-emerald-600 dark:text-emerald-400">
            FAQ ChantiFlow
          </p>
          <h2 className="mt-4 text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
            Tout ce qu’il faut savoir avant de déployer ChantiFlow
          </h2>
          <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
            Nous accompagnons chaque client avec un plan de mise en œuvre adapté à la taille de l’entreprise.
          </p>
        </div>
        <div className="mx-auto mt-12 max-w-4xl space-y-4">
          {faqEntries.map((entry) => (
            <div key={entry.question} className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">{entry.question}</h3>
              <p className="mt-2 text-sm leading-6 text-zinc-600 dark:text-zinc-400">{entry.answer}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto max-w-7xl px-6 py-12">
          <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
            {/* Logo et description */}
            <div className="col-span-1 md:col-span-2">
              <div className="flex items-center gap-2 mb-4">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-black text-white dark:bg-white dark:text-black">
                  <Zap className="h-5 w-5" />
                </div>
                <span className="text-lg font-bold text-zinc-900 dark:text-white">
                  ChantiFlow
                </span>
              </div>
              <p className="text-sm text-zinc-600 dark:text-zinc-400 max-w-md">
                La gestion de chantier simplifiée, assistée par l'intelligence artificielle.
              </p>
            </div>

            {/* Liens légaux */}
            <div>
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-white mb-4">
                Légal
              </h3>
              <ul className="space-y-2">
                <li>
                  <Link
                    href="/mentions-legales"
                    className="text-sm text-zinc-600 transition hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
                  >
                    Mentions légales
                  </Link>
                </li>
                <li>
                  <Link
                    href="/politique-confidentialite"
                    className="text-sm text-zinc-600 transition hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
                  >
                    Politique de confidentialité
                  </Link>
                </li>
              </ul>
            </div>

            {/* Contact */}
            <div>
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-white mb-4">
                Contact
              </h3>
              <ul className="space-y-2">
                <li>
                  <a
                    href="mailto:contact@chantiflow.com"
                    className="text-sm text-zinc-600 transition hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
                  >
                    contact@chantiflow.com
                  </a>
                </li>
                <li>
                  <a
                    href="mailto:admin@chantiflow.com"
                    className="text-sm text-zinc-600 transition hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
                  >
                    admin@chantiflow.com
                  </a>
                </li>
              </ul>
            </div>
          </div>

          <div className="mt-8 border-t border-zinc-200 pt-8 dark:border-zinc-800">
            <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                © 2025 ChantiFlow. Tous droits réservés.
              </p>
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                ChantiFlow SAS
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
