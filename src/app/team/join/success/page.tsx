import Link from 'next/link';
import { CheckCircle } from 'lucide-react';

export default function JoinTeamSuccessPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-100 via-white to-zinc-50 p-6 text-zinc-900 dark:from-zinc-900 dark:via-zinc-800 dark:to-black dark:text-white flex items-center justify-center">
      <div className="w-full max-w-md">
        <div className="rounded-3xl border border-zinc-200 bg-white/80 p-8 backdrop-blur dark:border-white/10 dark:bg-white/5 shadow-xl text-center">
          <div className="mb-6 flex justify-center">
            <div className="rounded-full bg-emerald-100 p-4 dark:bg-emerald-900/30">
              <CheckCircle className="h-12 w-12 text-emerald-600 dark:text-emerald-400" />
            </div>
          </div>
          
          <h1 className="text-3xl font-bold mb-4 dark:text-white">
            Demande envoyée !
          </h1>
          
          <p className="text-sm text-zinc-600 dark:text-zinc-400 mb-6">
            Votre demande d'ajout à l'équipe a bien été reçue.
          </p>
          
          <div className="rounded-lg bg-amber-50 border border-amber-200 p-4 mb-6 dark:bg-amber-900/20 dark:border-amber-800">
            <p className="text-sm text-amber-800 dark:text-amber-300 font-semibold mb-2">
              ⏳ En attente de validation
            </p>
            <p className="text-xs text-amber-700 dark:text-amber-400">
              Votre demande est actuellement en attente de validation par le chef de chantier. Vous recevrez un email de confirmation une fois votre demande approuvée.
            </p>
          </div>
          
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-6">
            En attendant, vous pouvez préparer vos informations et documents nécessaires pour votre intégration à l'équipe.
          </p>
          
          <Link
            href="/"
            className="inline-block rounded-lg bg-emerald-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-emerald-700"
          >
            Retour à l'accueil
          </Link>
        </div>
      </div>
    </div>
  );
}

