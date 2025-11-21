import { AppShell } from '@/components/app-shell';

export const metadata = {
  title: 'Politique de confidentialité | ChantiFlow',
  description: 'Politique de confidentialité de ChantiFlow',
};

export default function PolitiqueConfidentialitePage() {
  return (
    <AppShell heading="Politique de confidentialité" subheading="Protection de vos données" userEmail={null} primarySite={null}>
      <div className="mx-auto max-w-3xl">
        <div className="rounded-3xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div className="prose prose-zinc dark:prose-invert max-w-none">
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white mb-4">
              Collecte des données
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-6">
              ChantiFlow collecte uniquement les données nécessaires au fonctionnement du service : 
              informations de compte, données des chantiers, équipes et rapports.
            </p>

            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white mb-4">
              Utilisation des données
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-6">
              Vos données sont utilisées exclusivement pour fournir le service ChantiFlow. 
              Nous ne vendons pas vos données à des tiers.
            </p>

            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white mb-4">
              Sécurité
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-6">
              Nous mettons en œuvre des mesures de sécurité appropriées pour protéger vos données 
              contre tout accès non autorisé, perte ou destruction.
            </p>

            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white mb-4">
              Vos droits
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-6">
              Conformément au RGPD, vous disposez d'un droit d'accès, de rectification, 
              de suppression et de portabilité de vos données. Contactez-nous à contact@chantiflow.com.
            </p>

            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white mb-4">
              Cookies
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-6">
              Nous utilisons des cookies techniques nécessaires au fonctionnement du service. 
              Aucun cookie de tracking n'est utilisé.
            </p>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

