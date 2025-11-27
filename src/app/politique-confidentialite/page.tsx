import Link from 'next/link';
import { Footer } from '@/components/landing/footer';

export const metadata = {
  title: 'Politique de confidentialité | ChantiFlow',
  description: 'Politique de confidentialité de ChantiFlow - Protection de vos données personnelles',
};

export default function PolitiqueConfidentialitePage() {
  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950">
      <div className="mx-auto max-w-4xl px-6 py-16">
        <div className="mb-8">
          <Link
            href="/"
            className="text-sm text-emerald-600 hover:text-emerald-700 dark:text-emerald-400 dark:hover:text-emerald-300"
          >
            ← Retour à l&apos;accueil
          </Link>
        </div>

        <h1 className="mb-4 text-4xl font-bold text-zinc-900 dark:text-white">
          Politique de confidentialité
        </h1>
        <p className="mb-8 text-lg text-zinc-600 dark:text-zinc-400">
          ChantiFlow s&apos;engage à protéger votre vie privée et vos données personnelles. Cette politique explique comment
          nous collectons, utilisons et protégeons vos informations.
        </p>

        <div className="space-y-8 rounded-3xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              1. Collecte des données
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                ChantiFlow collecte uniquement les données nécessaires au fonctionnement du service :
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données de compte :</strong> nom, prénom, adresse email,
                  mot de passe (hashé)
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données de chantier :</strong> informations sur vos
                  chantiers, tâches, équipes et rapports
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données d&apos;utilisation :</strong> logs de connexion,
                  actions effectuées sur la plateforme
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données de paiement :</strong> informations de facturation
                  (gérées par Stripe, conformément à leurs politiques)
                </li>
              </ul>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              2. Utilisation des données
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>Vos données sont utilisées exclusivement pour :</p>
              <ul className="ml-6 list-disc space-y-2">
                <li>Fournir et améliorer le service ChantiFlow</li>
                <li>Gérer votre compte et vos préférences</li>
                <li>Vous contacter concernant votre compte ou le service</li>
                <li>Respecter nos obligations légales et réglementaires</li>
                <li>Prévenir la fraude et assurer la sécurité de la plateforme</li>
              </ul>
              <p className="mt-4">
                <strong className="text-zinc-900 dark:text-white">Important :</strong> Nous ne vendons jamais vos données à
                des tiers à des fins commerciales. Nous ne partageons vos données qu&apos;avec nos prestataires techniques
                nécessaires au fonctionnement du service.
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              3. Sécurité
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                Nous mettons en œuvre des mesures de sécurité appropriées pour protéger vos données contre tout accès non
                autorisé, perte ou destruction :
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>Chiffrement des données en transit (HTTPS/TLS)</li>
                <li>Chiffrement des données sensibles en base de données</li>
                <li>Authentification forte et gestion sécurisée des mots de passe</li>
                <li>Contrôle d&apos;accès basé sur les rôles</li>
                <li>Sauvegardes régulières et sécurisées</li>
                <li>Surveillance continue de la sécurité</li>
              </ul>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              4. Vos droits
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                Conformément au RGPD, vous disposez des droits suivants concernant vos données personnelles :
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-zinc-900 dark:text-white">Droit d&apos;accès :</strong> obtenir une copie de vos
                  données
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Droit de rectification :</strong> corriger vos données
                  inexactes
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Droit à l&apos;effacement :</strong> demander la suppression
                  de vos données
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Droit à la portabilité :</strong> récupérer vos données
                  dans un format structuré
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Droit d&apos;opposition :</strong> vous opposer au
                  traitement de vos données
                </li>
              </ul>
              <p className="mt-4">
                Pour exercer ces droits, contactez-nous à{' '}
                <a
                  href="mailto:contact@chantiflow.com"
                  className="text-emerald-600 hover:underline dark:text-emerald-400"
                >
                  contact@chantiflow.com
                </a>
                . Pour plus de détails, consultez notre{' '}
                <Link
                  href="/rgpd"
                  className="text-emerald-600 hover:underline dark:text-emerald-400"
                >
                  page RGPD
                </Link>
                .
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              5. Cookies
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                Nous utilisons uniquement des cookies techniques strictement nécessaires au fonctionnement du service :
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-zinc-900 dark:text-white">Cookies de session :</strong> pour maintenir votre
                  connexion
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Cookies de préférences :</strong> pour mémoriser vos
                  paramètres (thème sombre, etc.)
                </li>
              </ul>
              <p className="mt-4">
                <strong className="text-zinc-900 dark:text-white">Aucun cookie de tracking ou de publicité n&apos;est
                  utilisé.</strong> Nous ne suivons pas votre navigation sur d&apos;autres sites.
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              6. Partage de données
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>Vos données peuvent être partagées avec :</p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-zinc-900 dark:text-white">Prestataires techniques :</strong> Vercel (hébergement),
                  Supabase (base de données), Stripe (paiement), Resend (emails)
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Membres de votre équipe :</strong> uniquement les données
                  nécessaires à la collaboration sur vos chantiers
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Autorités compétentes :</strong> en cas d&apos;obligation
                  légale ou de réquisition judiciaire
                </li>
              </ul>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              7. Conservation des données
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                Vos données sont conservées pendant la durée nécessaire aux finalités pour lesquelles elles ont été collectées :
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données de compte :</strong> pendant toute la durée
                  d&apos;utilisation du service, puis 3 ans après la fermeture du compte
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données de chantier :</strong> pendant toute la durée
                  d&apos;activité, puis 10 ans pour les obligations légales
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données de facturation :</strong> 10 ans conformément aux
                  obligations comptables
                </li>
              </ul>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              8. Modifications
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400">
              Cette politique de confidentialité peut être modifiée à tout moment. Nous vous informerons de toute modification
              importante par email ou via une notification sur la plateforme.
            </p>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              9. Contact
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400">
              Pour toute question concernant cette politique de confidentialité, contactez-nous à{' '}
              <a
                href="mailto:contact@chantiflow.com"
                className="text-emerald-600 hover:underline dark:text-emerald-400"
              >
                contact@chantiflow.com
              </a>
              .
            </p>
          </section>

          <div className="mt-8 border-t border-zinc-200 pt-6 dark:border-zinc-800">
            <p className="text-sm text-zinc-500 dark:text-zinc-500">
              Dernière mise à jour : {new Date().toLocaleDateString('fr-FR', { year: 'numeric', month: 'long', day: 'numeric' })}
            </p>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}


