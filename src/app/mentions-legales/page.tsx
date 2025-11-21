import { AppShell } from '@/components/app-shell';

export const metadata = {
  title: 'Mentions légales | ChantiFlow',
  description: 'Mentions légales de ChantiFlow',
};

export default function MentionsLegalesPage() {
  return (
    <AppShell heading="Mentions légales" subheading="Informations légales" userEmail={null} primarySite={null}>
      <div className="mx-auto max-w-3xl">
        <div className="rounded-3xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div className="prose prose-zinc dark:prose-invert max-w-none">
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white mb-4">
              Éditeur
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-6">
              ChantiFlow SAS<br />
              Email : contact@chantiflow.com<br />
              Support : admin@chantiflow.com
            </p>

            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white mb-4">
              Directeur de publication
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-6">
              Les fondateurs de ChantiFlow
            </p>

            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white mb-4">
              Hébergement
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-6">
              Vercel Inc.<br />
              Base de données : Supabase
            </p>

            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white mb-4">
              Données personnelles
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-6">
              Les données personnelles collectées sont utilisées uniquement pour le fonctionnement du service. 
              Conformément au RGPD, vous disposez d'un droit d'accès, de rectification et de suppression de vos données.
            </p>

            <h2 className="text-xl font-semibold text-zinc-900 dark:text-white mb-4">
              Propriété intellectuelle
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-6">
              L'ensemble du contenu de ce site (textes, images, logos) est la propriété de ChantiFlow SAS 
              et est protégé par les lois relatives à la propriété intellectuelle.
            </p>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

