import Link from 'next/link';
import { Check, Zap, Users, QrCode, Brain, Calendar, Clock, AlertTriangle, MessageSquare, FileText, Camera, TrendingUp, Shield } from 'lucide-react';
import { PricingSection } from '@/components/pricing-section';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export const metadata = {
  title: 'ChantiFlow - Gestion de chantiers simplifiée avec IA',
  description: 'Gérez vos chantiers efficacement avec planification IA, QR codes pour les employés, et suivi en temps réel.',
};

export default async function LandingPage() {
  // Vérifier si l'utilisateur est connecté
  let isAuthenticated = false;
  let userEmail: string | null = null;
  try {
    const supabase = await createSupabaseServerClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();
    isAuthenticated = !!user;
    userEmail = user?.email || null;
  } catch {
    isAuthenticated = false;
  }
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
            Gérez vos chantiers{' '}
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

      {/* Pricing Section */}
      <PricingSection isAuthenticated={isAuthenticated} userEmail={userEmail} />

      {/* À propos */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-3xl">
          <div className="rounded-3xl border border-zinc-200 bg-white p-12 text-center dark:border-zinc-800 dark:bg-zinc-900">
            <h2 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
              Qui sommes-nous ?
            </h2>
            <p className="mt-6 text-lg leading-8 text-zinc-600 dark:text-zinc-400">
              Nous sommes deux jeunes entrepreneurs passionnés par la modernisation du BTP. 
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
