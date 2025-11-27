import Link from 'next/link';
import { Footer } from '@/components/landing/footer';

export const metadata = {
  title: 'Mentions légales | ChantiFlow',
  description: 'Mentions légales de ChantiFlow - Informations légales et conditions d\'utilisation',
};

export default function MentionsLegalesPage() {
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

        <h1 className="mb-8 text-4xl font-bold text-zinc-900 dark:text-white">
          Mentions légales
        </h1>

        <div className="space-y-8 rounded-3xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              1. Éditeur du site
            </h2>
            <div className="space-y-2 text-zinc-600 dark:text-zinc-400">
              <p>
                <strong className="text-zinc-900 dark:text-white">Raison sociale :</strong> ChantiFlow SAS
              </p>
              <p>
                <strong className="text-zinc-900 dark:text-white">Forme juridique :</strong> Société par Actions Simplifiée
              </p>
              <p>
                <strong className="text-zinc-900 dark:text-white">Email :</strong>{' '}
                <a
                  href="mailto:contact@chantiflow.com"
                  className="text-emerald-600 hover:underline dark:text-emerald-400"
                >
                  contact@chantiflow.com
                </a>
              </p>
              <p>
                <strong className="text-zinc-900 dark:text-white">Support technique :</strong>{' '}
                <a
                  href="mailto:admin@chantiflow.com"
                  className="text-emerald-600 hover:underline dark:text-emerald-400"
                >
                  admin@chantiflow.com
                </a>
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              2. Directeur de publication
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400">
              Le directeur de publication est représenté par les fondateurs de ChantiFlow SAS.
            </p>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              3. Hébergement
            </h2>
            <div className="space-y-2 text-zinc-600 dark:text-zinc-400">
              <p>
                <strong className="text-zinc-900 dark:text-white">Hébergeur web :</strong> Vercel Inc.
              </p>
              <p>
                <strong className="text-zinc-900 dark:text-white">Adresse :</strong> 340 S Lemon Ave #4133, Walnut, CA 91789, États-Unis
              </p>
              <p>
                <strong className="text-zinc-900 dark:text-white">Base de données :</strong> Supabase (hébergement UE)
              </p>
              <p>
                <strong className="text-zinc-900 dark:text-white">Site web :</strong>{' '}
                <a
                  href="https://vercel.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-emerald-600 hover:underline dark:text-emerald-400"
                >
                  vercel.com
                </a>
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              4. Conditions d&apos;utilisation
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                L&apos;utilisation du site ChantiFlow implique l&apos;acceptation pleine et entière des conditions générales
                d&apos;utilisation ci-après décrites. Ces conditions d&apos;utilisation sont susceptibles d&apos;être modifiées ou
                complétées à tout moment.
              </p>
              <p>
                Les utilisateurs du site sont donc invités à les consulter de manière régulière. Le site est normalement
                accessible à tout moment aux utilisateurs. Une interruption pour raison de maintenance technique peut être
                toutefois décidée par ChantiFlow.
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              5. Données personnelles
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                Conformément aux dispositions de la loi n° 78-17 du 6 janvier 1978 modifiée relative à l&apos;informatique, aux
                fichiers et aux libertés et du Règlement Général sur la Protection des Données (RGPD), vous disposez d&apos;un
                droit d&apos;accès, de rectification, de suppression et de portabilité de vos données personnelles.
              </p>
              <p>
                Pour exercer ces droits, vous pouvez nous contacter à l&apos;adresse{' '}
                <a
                  href="mailto:contact@chantiflow.com"
                  className="text-emerald-600 hover:underline dark:text-emerald-400"
                >
                  contact@chantiflow.com
                </a>
                .
              </p>
              <p>
                Pour plus d&apos;informations, consultez notre{' '}
                <Link
                  href="/rgpd"
                  className="text-emerald-600 hover:underline dark:text-emerald-400"
                >
                  page dédiée au RGPD
                </Link>
                {' '}et notre{' '}
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
              6. Propriété intellectuelle
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                L&apos;ensemble du contenu de ce site (textes, images, logos, icônes, vidéos, graphismes) est la propriété
                exclusive de ChantiFlow SAS, sauf mention contraire.
              </p>
              <p>
                Toute reproduction, représentation, modification, publication, adaptation de tout ou partie des éléments du
                site, quel que soit le moyen ou le procédé utilisé, est interdite, sauf autorisation écrite préalable de
                ChantiFlow SAS.
              </p>
              <p>
                Toute exploitation non autorisée du site ou de l&apos;un quelconque des éléments qu&apos;il contient sera
                considérée comme constitutive d&apos;une contrefaçon et poursuivie conformément aux dispositions des articles
                L.335-2 et suivants du Code de la Propriété Intellectuelle.
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              7. Responsabilité
            </h2>
            <div className="space-y-4 text-zinc-600 dark:text-zinc-400">
              <p>
                ChantiFlow SAS ne pourra être tenu responsable des dommages directs et indirects causés au matériel de
                l&apos;utilisateur, lors de l&apos;accès au site ChantiFlow, et résultant soit de l&apos;utilisation d&apos;un
                matériel ne répondant pas aux spécifications, soit de l&apos;apparition d&apos;un bug ou d&apos;une
                incompatibilité.
              </p>
              <p>
                ChantiFlow SAS ne pourra également être tenu responsable des dommages indirects consécutifs à
                l&apos;utilisation du site.
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              8. Droit applicable et juridiction compétente
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400">
              Tout litige en relation avec l&apos;utilisation du site ChantiFlow est soumis au droit français. Il est fait
              attribution exclusive de juridiction aux tribunaux compétents de Paris.
            </p>
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold text-zinc-900 dark:text-white">
              9. Contact
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400">
              Pour toute question relative aux présentes mentions légales, vous pouvez nous contacter à l&apos;adresse{' '}
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

