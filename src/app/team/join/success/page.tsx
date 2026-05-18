import Link from 'next/link';
import { CheckCircle } from 'lucide-react';

export default function JoinTeamSuccessPage() {
  return (
    <div className="min-h-screen bg-paper p-6 text-ink flex items-center justify-center dark:bg-ink dark:text-paper">
      <div className="w-full max-w-md">
        <div className="border border-rule-soft bg-paper p-8 text-center dark:border-rule dark:bg-ink">
          <div className="mb-6 flex justify-center">
            <div className="bg-paper-2 p-4">
              <CheckCircle className="h-12 w-12 text-orange dark:text-green" />
            </div>
          </div>

          <h1 className="font-serif text-[32px] text-ink mb-4 dark:text-paper">
            Demande envoyée !
          </h1>

          <p className="text-sm text-ink-2 mb-6">
            Votre demande d&apos;ajout à l&apos;équipe a bien été reçue.
          </p>

          <div className="border border-warn bg-paper-2 p-4 mb-6">
            <p className="font-mono text-[10px] uppercase tracking-widest text-warn font-semibold mb-2">
              ⏳ En attente de validation
            </p>
            <p className="text-xs text-ink-2">
              Votre demande est actuellement en attente de validation par le chef de chantier. Vous recevrez un email de confirmation une fois votre demande approuvée.
            </p>
          </div>

          <p className="text-sm text-ink-3 mb-6">
            En attendant, vous pouvez préparer vos informations et documents nécessaires pour votre intégration à l&apos;équipe.
          </p>

          <Link
            href="/"
            className="inline-block border border-orange bg-orange px-6 py-3 font-mono text-[11px] uppercase tracking-widest text-paper transition hover:bg-orange-dark"
          >
            Retour à l&apos;accueil
          </Link>
        </div>
      </div>
    </div>
  );
}
