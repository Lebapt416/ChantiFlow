import Link from 'next/link';
import { Footer } from '@/components/landing/footer';

export const metadata = {
  title: 'RGPD - Protection des données | ChantiFlow',
  description: 'Conformité RGPD de ChantiFlow - Vos droits et notre engagement pour la protection de vos données personnelles',
};

export default function RGPDPage() {
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
          Conformité RGPD
        </h1>
        <p className="mb-8 text-lg text-zinc-600 dark:text-zinc-400">
          ChantiFlow s&apos;engage à respecter le Règlement Général sur la Protection des Données (RGPD) et à protéger vos
          données personnelles.
        </p>

        <div className="space-y-8 rounded-3xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              1. Responsable du traitement
            </h2>
            <div className="space-y-2 text-zinc-600 dark:text-zinc-400">
              <p>
                <strong className="text-zinc-900 dark:text-white">Responsable :</strong> ChantiFlow SAS
              </p>
              <p>
                <strong className="text-zinc-900 dark:text-white">Contact :</strong>{' '}
                <a
                  href="mailto:contact@chantiflow.com"
                  className="text-emerald-600 hover:underline dark:text-emerald-400"
                >
                  contact@chantiflow.com
                </a>
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              2. Données collectées
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                ChantiFlow collecte et traite les données personnelles suivantes dans le cadre de l&apos;utilisation de la
                plateforme :
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données d&apos;identification :</strong> nom, prénom,
                  adresse email
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données de connexion :</strong> identifiants de connexion
                  (hashés)
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données professionnelles :</strong> rôle, métier, équipe
                  d&apos;appartenance
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données de chantier :</strong> informations sur les
                  chantiers, tâches, rapports, photos
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données de navigation :</strong> cookies techniques
                  nécessaires au fonctionnement du service
                </li>
              </ul>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              3. Finalités du traitement
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>Vos données personnelles sont utilisées pour les finalités suivantes :</p>
              <ul className="ml-6 list-disc space-y-2">
                <li>Gestion de votre compte utilisateur</li>
                <li>Fourniture du service de gestion de chantiers</li>
                <li>Planification et suivi des tâches</li>
                <li>Communication avec les membres de votre équipe</li>
                <li>Génération de rapports et statistiques</li>
                <li>Amélioration de nos services</li>
                <li>Respect de nos obligations légales et réglementaires</li>
              </ul>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              4. Base légale du traitement
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>Le traitement de vos données personnelles est fondé sur :</p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-zinc-900 dark:text-white">L&apos;exécution d&apos;un contrat :</strong> pour la
                  fourniture du service ChantiFlow
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Votre consentement :</strong> pour l&apos;envoi de
                  communications marketing (si applicable)
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">L&apos;intérêt légitime :</strong> pour l&apos;amélioration
                  de nos services et la prévention de la fraude
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Les obligations légales :</strong> pour le respect de nos
                  obligations comptables et fiscales
                </li>
              </ul>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              5. Conservation des données
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                Vos données personnelles sont conservées pendant la durée nécessaire aux finalités pour lesquelles elles ont
                été collectées, et conformément aux obligations légales applicables :
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données de compte :</strong> pendant toute la durée
                  d&apos;utilisation du service, puis 3 ans après la fermeture du compte
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Données de chantier :</strong> pendant toute la durée
                  d&apos;activité du chantier, puis 10 ans pour les obligations légales
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
              6. Destinataires des données
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>Vos données personnelles peuvent être communiquées aux destinataires suivants :</p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-zinc-900 dark:text-white">Personnel autorisé de ChantiFlow :</strong> pour la
                  gestion et le support du service
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Prestataires techniques :</strong> hébergement (Vercel,
                  Supabase), services de paiement (Stripe), services d&apos;email (Resend)
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Autorités compétentes :</strong> en cas d&apos;obligation
                  légale ou de réquisition judiciaire
                </li>
              </ul>
              <p className="mt-4">
                <strong className="text-zinc-900 dark:text-white">Important :</strong> Nous ne vendons jamais vos données
                personnelles à des tiers à des fins commerciales.
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              7. Transferts de données hors UE
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                Vos données sont principalement hébergées dans l&apos;Union Européenne (Supabase). Certains de nos
                prestataires peuvent être situés hors de l&apos;UE :
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-zinc-900 dark:text-white">Vercel (hébergement) :</strong> États-Unis (clauses
                  contractuelles types approuvées par la Commission européenne)
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Stripe (paiement) :</strong> États-Unis (Privacy Shield
                  et clauses contractuelles)
                </li>
              </ul>
              <p className="mt-4">
                Ces transferts sont encadrés par des garanties appropriées conformes au RGPD.
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              8. Vos droits
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>Conformément au RGPD, vous disposez des droits suivants :</p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-zinc-900 dark:text-white">Droit d&apos;accès :</strong> vous pouvez obtenir une copie
                  de vos données personnelles
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Droit de rectification :</strong> vous pouvez corriger vos
                  données inexactes ou incomplètes
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Droit à l&apos;effacement :</strong> vous pouvez demander
                  la suppression de vos données dans certains cas
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Droit à la limitation :</strong> vous pouvez demander la
                  limitation du traitement de vos données
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Droit à la portabilité :</strong> vous pouvez récupérer vos
                  données dans un format structuré
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Droit d&apos;opposition :</strong> vous pouvez vous
                  opposer au traitement de vos données pour des motifs légitimes
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Droit de retrait du consentement :</strong> lorsque le
                  traitement est basé sur votre consentement
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
                {' '}en précisant votre demande et en joignant une copie de votre pièce d&apos;identité.
              </p>
              <p>
                Vous disposez également du droit d&apos;introduire une réclamation auprès de la CNIL (Commission Nationale de
                l&apos;Informatique et des Libertés) si vous estimez que vos droits ne sont pas respectés.
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              9. Sécurité des données
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                ChantiFlow met en œuvre des mesures techniques et organisationnelles appropriées pour protéger vos données
                personnelles contre tout accès non autorisé, perte, destruction ou altération :
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>Chiffrement des données en transit (HTTPS/TLS)</li>
                <li>Chiffrement des données sensibles en base</li>
                <li>Authentification forte et gestion sécurisée des mots de passe</li>
                <li>Contrôle d&apos;accès basé sur les rôles</li>
                <li>Sauvegardes régulières et sécurisées</li>
                <li>Surveillance et détection des incidents de sécurité</li>
              </ul>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              10. Cookies
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                ChantiFlow utilise uniquement des cookies techniques strictement nécessaires au fonctionnement du service. Aucun
                cookie de tracking ou de publicité n&apos;est utilisé.
              </p>
              <p>
                Pour plus d&apos;informations, consultez notre{' '}
                <Link
                  href="/politique-confidentialite"
                  className="text-emerald-600 hover:underline dark:text-emerald-400"
                >
                  politique de confidentialité
                </Link>
                .
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              11. Contact et DPO
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                Pour toute question relative au traitement de vos données personnelles ou pour exercer vos droits, vous pouvez
                nous contacter :
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-zinc-900 dark:text-white">Email :</strong>{' '}
                  <a
                    href="mailto:contact@chantiflow.com"
                    className="text-emerald-600 hover:underline dark:text-emerald-400"
                  >
                    contact@chantiflow.com
                  </a>
                </li>
                <li>
                  <strong className="text-zinc-900 dark:text-white">Support :</strong>{' '}
                  <a
                    href="mailto:admin@chantiflow.com"
                    className="text-emerald-600 hover:underline dark:text-emerald-400"
                  >
                    admin@chantiflow.com
                  </a>
                </li>
              </ul>
            </div>
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

