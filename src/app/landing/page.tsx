import Link from 'next/link';
import { Check, Zap, Users, QrCode, Brain, Calendar } from 'lucide-react';
import { PricingSection } from '@/components/pricing-section';

export const metadata = {
  title: 'ChantiFlow - Gestion de chantiers simplifiée avec IA',
  description: 'Gérez vos chantiers efficacement avec planification IA, QR codes pour les employés, et suivi en temps réel.',
};

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950">
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
          <h1 className="text-5xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-6xl">
            Gérez vos chantiers avec{' '}
            <span className="text-emerald-600 dark:text-emerald-400">l'intelligence artificielle</span>
          </h1>
          <p className="mt-6 text-lg leading-8 text-zinc-600 dark:text-zinc-400">
            ChantiFlow simplifie la gestion de vos chantiers avec planification automatique, QR codes
            pour vos équipes, et suivi en temps réel. Optimisez vos projets de construction dès
            aujourd'hui.
          </p>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <Link
              href="/login"
              className="rounded-full bg-black px-6 py-3 text-base font-semibold text-white shadow-lg transition hover:bg-zinc-800 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
            >
              Démarrer gratuitement
            </Link>
            <Link
              href="#features"
              className="text-base font-semibold leading-6 text-zinc-900 dark:text-white"
            >
              En savoir plus <span aria-hidden="true">→</span>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
            Tout ce dont vous avez besoin pour gérer vos chantiers
          </h2>
          <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
            Des outils puissants pour optimiser vos projets de construction
          </p>
        </div>

        <div className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-8 sm:grid-cols-2 lg:max-w-none lg:grid-cols-3">
          <div className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-emerald-100 dark:bg-emerald-900/30">
              <Brain className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
            </div>
            <h3 className="mt-6 text-lg font-semibold text-zinc-900 dark:text-white">
              Planification IA
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Notre IA génère automatiquement un planning optimisé en fonction de vos tâches, équipes
              et deadlines. Les tâches sont classées par ordre logique et priorité.
            </p>
          </div>

          <div className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/30">
              <QrCode className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
            <h3 className="mt-6 text-lg font-semibold text-zinc-900 dark:text-white">
              Accès QR codes
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Vos employés accèdent à leurs tâches via QR code, sans compte. Ils peuvent uploader
              des photos et envoyer des rapports directement depuis leur téléphone.
            </p>
          </div>

          <div className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900/30">
              <Users className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <h3 className="mt-6 text-lg font-semibold text-zinc-900 dark:text-white">
              Gestion d'équipe
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Ajoutez vos équipes, assignez des rôles, et suivez l'avancement de chaque projet en
              temps réel avec des statistiques détaillées.
            </p>
          </div>

          <div className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-100 dark:bg-amber-900/30">
              <Calendar className="h-6 w-6 text-amber-600 dark:text-amber-400" />
            </div>
            <h3 className="mt-6 text-lg font-semibold text-zinc-900 dark:text-white">
              Suivi en temps réel
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Visualisez l'avancement de vos chantiers, les tâches terminées, et recevez des
              rapports avec photos directement depuis le terrain.
            </p>
          </div>

          <div className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-rose-100 dark:bg-rose-900/30">
              <Zap className="h-6 w-6 text-rose-600 dark:text-rose-400" />
            </div>
            <h3 className="mt-6 text-lg font-semibold text-zinc-900 dark:text-white">
              Interface simple
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Une interface intuitive et moderne, accessible depuis n'importe quel appareil. Mode
              sombre inclus pour travailler dans toutes les conditions.
            </p>
          </div>

          <div className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-indigo-100 dark:bg-indigo-900/30">
              <Check className="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
            </div>
            <h3 className="mt-6 text-lg font-semibold text-zinc-900 dark:text-white">
              Optimisation automatique
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              L'IA classe automatiquement vos tâches par ordre logique, identifie les dépendances,
              et optimise l'utilisation de vos ressources.
            </p>
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <PricingSection />

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
              className="rounded-full bg-white px-6 py-3 text-base font-semibold text-zinc-900 shadow-lg transition hover:bg-zinc-100"
            >
              Commencer gratuitement
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-zinc-200 dark:border-zinc-800">
        <div className="mx-auto max-w-7xl px-6 py-12">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-black text-white dark:bg-white dark:text-black">
                <Zap className="h-5 w-5" />
              </div>
              <span className="text-lg font-bold text-zinc-900 dark:text-white">
                ChantiFlow
              </span>
            </div>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              © 2025 ChantiFlow. Tous droits réservés.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

