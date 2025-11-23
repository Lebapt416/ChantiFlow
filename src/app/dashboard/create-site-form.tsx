"use client";

import { useActionState, useEffect, useState } from "react";
import { useFormStatus } from "react-dom";
import { useRouter } from "next/navigation";
import { Sparkles, Loader2, Calendar } from "lucide-react";
import { createSiteAction, type CreateSiteState } from './actions';
import { getPrediction } from '@/lib/ai/prediction';

const initialState: CreateSiteState = {};

function SubmitButton() {
  const { pending } = useFormStatus();

  return (
    <button
      type="submit"
      className="rounded-md bg-black px-4 py-2 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-70 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
      disabled={pending}
    >
      {pending ? 'Création...' : 'Créer'}
    </button>
  );
}

type Props = {
  onSuccess?: () => void;
};

export function CreateSiteForm({ onSuccess }: Props) {
  const router = useRouter();
  const [state, formAction] = useActionState(createSiteAction, initialState);
  const [nombreTaches, setNombreTaches] = useState<string>('');
  const [complexite, setComplexite] = useState<string>('5');
  const [predictedDuree, setPredictedDuree] = useState<number | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  useEffect(() => {
    if (state?.success) {
      const form = document.getElementById('create-site-form') as HTMLFormElement | null;
      form?.reset();
      setNombreTaches('');
      setComplexite('5');
      setPredictedDuree(null);
      if (onSuccess) {
        onSuccess();
      } else {
        router.push('/sites');
        router.refresh();
      }
    }
  }, [state?.success, router, onSuccess]);

  // Fonction pour obtenir la prédiction
  const handlePredict = async () => {
    const taches = parseInt(nombreTaches);
    const comp = parseFloat(complexite);

    if (!taches || taches < 1 || taches > 100) {
      setPredictionError('Le nombre de tâches doit être entre 1 et 100');
      return;
    }

    if (!comp || comp < 1 || comp > 10) {
      setPredictionError('La complexité doit être entre 1.0 et 10.0');
      return;
    }

    setIsPredicting(true);
    setPredictionError(null);

    try {
      const duree = await getPrediction(taches, comp);
      setPredictedDuree(duree);
      
      // Mettre à jour automatiquement la deadline si elle n'est pas définie
      const deadlineInput = document.getElementById('deadline') as HTMLInputElement;
      if (deadlineInput && !deadlineInput.value) {
        const today = new Date();
        today.setDate(today.getDate() + Math.ceil(duree));
        deadlineInput.value = today.toISOString().split('T')[0];
      }
    } catch (error) {
      setPredictionError(
        error instanceof Error 
          ? error.message 
          : 'Erreur lors de la prédiction. Vérifiez que l\'API est démarrée.'
      );
      setPredictedDuree(null);
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <form
      id="create-site-form"
      action={formAction}
      className="space-y-4 rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900"
    >
      {/* Nom du chantier */}
      <div className="space-y-2">
        <label
          htmlFor="name"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Nom du chantier
        </label>
        <input
          id="name"
          name="name"
          placeholder="Résidence Soleil"
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          required
        />
      </div>

      {/* Section IA de prédiction */}
      <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 dark:border-emerald-800 dark:bg-emerald-900/20">
        <div className="mb-3 flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
          <h3 className="text-sm font-semibold text-emerald-900 dark:text-emerald-100">
            Prédiction IA de la durée
          </h3>
        </div>
        
        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-2">
            <label
              htmlFor="nombre_taches"
              className="text-xs font-medium text-emerald-800 dark:text-emerald-200"
            >
              Nombre de tâches estimé
            </label>
            <input
              id="nombre_taches"
              type="number"
              min="1"
              max="100"
              value={nombreTaches}
              onChange={(e) => setNombreTaches(e.target.value)}
              placeholder="20"
              className="w-full rounded-md border border-emerald-200 bg-white px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-emerald-700 dark:bg-zinc-950 dark:text-white"
            />
          </div>
          
          <div className="space-y-2">
            <label
              htmlFor="complexite"
              className="text-xs font-medium text-emerald-800 dark:text-emerald-200"
            >
              Complexité (1-10)
            </label>
            <div className="flex gap-2">
              <input
                id="complexite"
                type="number"
                min="1"
                max="10"
                step="0.1"
                value={complexite}
                onChange={(e) => setComplexite(e.target.value)}
                placeholder="5.0"
                className="flex-1 rounded-md border border-emerald-200 bg-white px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-emerald-700 dark:bg-zinc-950 dark:text-white"
              />
              <button
                type="button"
                onClick={handlePredict}
                disabled={isPredicting || !nombreTaches || !complexite}
                className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isPredicting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Résultat de la prédiction */}
        {predictedDuree !== null && (
          <div className="mt-3 flex items-center gap-2 rounded-md bg-white p-3 dark:bg-zinc-800">
            <Calendar className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
            <div className="flex-1">
              <p className="text-xs text-emerald-700 dark:text-emerald-300">
                Durée estimée par l'IA
              </p>
              <p className="text-sm font-semibold text-emerald-900 dark:text-emerald-100">
                {predictedDuree.toFixed(1)} jours
              </p>
            </div>
          </div>
        )}

        {/* Erreur de prédiction */}
        {predictionError && (
          <div className="mt-3 rounded-md bg-rose-100 p-2 text-xs text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">
            {predictionError}
          </div>
        )}
      </div>

      {/* Deadline */}
      <div className="space-y-2">
        <label
          htmlFor="deadline"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Deadline
          {predictedDuree && (
            <span className="ml-2 text-xs text-emerald-600 dark:text-emerald-400">
              (Suggérée: {new Date(Date.now() + predictedDuree * 24 * 60 * 60 * 1000).toLocaleDateString('fr-FR')})
            </span>
          )}
        </label>
        <input
          id="deadline"
          name="deadline"
          type="date"
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          required
        />
      </div>

      {/* Bouton de soumission */}
      <div className="flex items-center justify-between pt-2">
        <div className="text-xs text-zinc-500 dark:text-zinc-400">
          {state?.error && (
            <span className="text-rose-600 dark:text-rose-400">{state.error}</span>
          )}
          {state?.success && (
            <span className="text-emerald-600 dark:text-emerald-400">
              Chantier créé avec succès.
            </span>
          )}
        </div>
        <SubmitButton />
      </div>
    </form>
  );
}

