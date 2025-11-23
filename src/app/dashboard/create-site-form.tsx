"use client";

import { useActionState, useEffect, useState } from "react";
import { useFormStatus } from "react-dom";
import { useRouter } from "next/navigation";
import { Sparkles, Loader2, Calendar, AlertTriangle, Users } from "lucide-react";
import { createSiteAction, type CreateSiteState } from './actions';
import { getPrediction } from '@/lib/ai/prediction';
import { analyserRisqueRetard } from '@/lib/ai/risk-analysis';
import { recommanderEquipe } from '@/lib/ai/team-recommendation';

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
  
  // État pour l'analyse de risque
  const [risqueRetard, setRisqueRetard] = useState<{ pourcentage: number; justification: string } | null>(null);
  const [isAnalyzingRisk, setIsAnalyzingRisk] = useState(false);
  const [riskError, setRiskError] = useState<string | null>(null);
  
  // État pour la recommandation d'équipe
  const [typeTache, setTypeTache] = useState<string>('');
  const [recommandationEquipe, setRecommandationEquipe] = useState<{ equipe: string; performance: number } | null>(null);
  const [isRecommending, setIsRecommending] = useState(false);
  const [recommendationError, setRecommendationError] = useState<string | null>(null);

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
      const errorMessage = error instanceof Error 
        ? error.message 
        : 'Erreur lors de la prédiction.';
      
      // Message plus clair si l'API n'est pas accessible
      if (errorMessage.includes('Impossible de se connecter') || errorMessage.includes('fetch')) {
        setPredictionError(
          'API non accessible. Assurez-vous que le serveur FastAPI est démarré (port 8000). ' +
          'Exécutez: cd ml && python api.py'
        );
      } else {
        setPredictionError(errorMessage);
      }
      setPredictedDuree(null);
    } finally {
      setIsPredicting(false);
    }
  };

  // Fonction pour analyser le risque de retard
  const handleAnalyzeRisk = async () => {
    const comp = parseFloat(complexite);
    const taches = parseInt(nombreTaches);

    if (!comp || comp < 1 || comp > 10) {
      setRiskError('La complexité doit être entre 1.0 et 10.0');
      return;
    }

    setIsAnalyzingRisk(true);
    setRiskError(null);

    try {
      // Simuler un historique basé sur le nombre de tâches et la complexité
      // En production, cela viendrait de la base de données
      const historiqueSimule = Array.from({ length: Math.min(10, Math.max(3, Math.floor(taches / 5))) }, () => {
        const base = taches * 0.5;
        const variation = (comp / 10) * base * 0.3;
        return Math.max(1, Math.round(base + (Math.random() - 0.5) * variation));
      });

      const result = await analyserRisqueRetard(historiqueSimule, Math.round(comp));
      setRisqueRetard({
        pourcentage: result.risque_pourcentage,
        justification: result.justification,
      });
    } catch (error) {
      const errorMessage = error instanceof Error
        ? error.message
        : 'Erreur lors de l\'analyse de risque.';
      
      // Message plus clair si l'API n'est pas accessible
      if (errorMessage.includes('Impossible de se connecter') || errorMessage.includes('fetch')) {
        setRiskError(
          'API non accessible. Assurez-vous que le serveur FastAPI est démarré (port 8000). ' +
          'Exécutez: cd ml && python api.py'
        );
      } else {
        setRiskError(errorMessage);
      }
      setRisqueRetard(null);
    } finally {
      setIsAnalyzingRisk(false);
    }
  };

  // Fonction pour recommander une équipe
  const handleRecommendTeam = async () => {
    if (!typeTache || typeTache.trim() === '') {
      setRecommendationError('Veuillez entrer un type de tâche');
      return;
    }

    const comp = parseFloat(complexite);
    if (!comp || comp < 1 || comp > 10) {
      setRecommendationError('La complexité doit être entre 1.0 et 10.0');
      return;
    }

    setIsRecommending(true);
    setRecommendationError(null);

    try {
      const result = await recommanderEquipe(typeTache, Math.round(comp));
      setRecommandationEquipe({
        equipe: result.equipe_recommandee,
        performance: result.performance_attendue,
      });
    } catch (error) {
      const errorMessage = error instanceof Error
        ? error.message
        : 'Erreur lors de la recommandation d\'équipe.';
      
      // Message plus clair si l'API n'est pas accessible
      if (errorMessage.includes('Impossible de se connecter') || errorMessage.includes('fetch')) {
        setRecommendationError(
          'API non accessible. Assurez-vous que le serveur FastAPI est démarré (port 8000). ' +
          'Exécutez: cd ml && python api.py'
        );
      } else {
        setRecommendationError(errorMessage);
      }
      setRecommandationEquipe(null);
    } finally {
      setIsRecommending(false);
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

      {/* Section Analyse de risque */}
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-900/20">
        <div className="mb-3 flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-400" />
          <h3 className="text-sm font-semibold text-amber-900 dark:text-amber-100">
            Analyse de risque de retard
          </h3>
        </div>
        
        <button
          type="button"
          onClick={handleAnalyzeRisk}
          disabled={isAnalyzingRisk || !complexite}
          className="w-full rounded-md bg-amber-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-amber-700 disabled:cursor-not-allowed disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {isAnalyzingRisk ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Analyse en cours...
            </>
          ) : (
            <>
              <AlertTriangle className="h-4 w-4" />
              Vérifier le risque de retard
            </>
          )}
        </button>

        {/* Résultat de l'analyse de risque */}
        {risqueRetard && (
          <div className="mt-3 rounded-md bg-white p-3 dark:bg-zinc-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-amber-800 dark:text-amber-200">
                Risque de retard
              </span>
              <span
                className={`text-lg font-bold ${
                  risqueRetard.pourcentage >= 70
                    ? 'text-rose-600 dark:text-rose-400'
                    : risqueRetard.pourcentage >= 40
                    ? 'text-amber-600 dark:text-amber-400'
                    : 'text-emerald-600 dark:text-emerald-400'
                }`}
              >
                {risqueRetard.pourcentage}%
              </span>
            </div>
            <p className="text-xs text-amber-700 dark:text-amber-300 mt-1">
              {risqueRetard.justification}
            </p>
          </div>
        )}

        {/* Erreur d'analyse de risque */}
        {riskError && (
          <div className="mt-3 rounded-md bg-rose-100 p-2 text-xs text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">
            {riskError}
          </div>
        )}
      </div>

      {/* Section Recommandation d'équipe */}
      <div className="rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
        <div className="mb-3 flex items-center gap-2">
          <Users className="h-5 w-5 text-blue-600 dark:text-blue-400" />
          <h3 className="text-sm font-semibold text-blue-900 dark:text-blue-100">
            Recommandation d'équipe
          </h3>
        </div>
        
        <div className="space-y-2">
          <input
            type="text"
            value={typeTache}
            onChange={(e) => setTypeTache(e.target.value)}
            placeholder="Ex: maçonnerie, plomberie, électricité..."
            className="w-full rounded-md border border-blue-200 bg-white px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:border-blue-700 dark:bg-zinc-950 dark:text-white"
          />
          <button
            type="button"
            onClick={handleRecommendTeam}
            disabled={isRecommending || !typeTache || !complexite}
            className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {isRecommending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Recherche...
              </>
            ) : (
              <>
                <Users className="h-4 w-4" />
                Recommander une équipe
              </>
            )}
          </button>
        </div>

        {/* Résultat de la recommandation */}
        {recommandationEquipe && (
          <div className="mt-3 rounded-md bg-white p-3 dark:bg-zinc-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-blue-800 dark:text-blue-200">
                Équipe recommandée
              </span>
              <span className="text-sm font-bold text-blue-900 dark:text-blue-100">
                {recommandationEquipe.equipe}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-blue-700 dark:text-blue-300">
                Performance attendue:
              </span>
              <span className="text-sm font-semibold text-blue-900 dark:text-blue-100">
                {recommandationEquipe.performance.toFixed(1)}/10
              </span>
            </div>
          </div>
        )}

        {/* Erreur de recommandation */}
        {recommendationError && (
          <div className="mt-3 rounded-md bg-rose-100 p-2 text-xs text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">
            {recommendationError}
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

