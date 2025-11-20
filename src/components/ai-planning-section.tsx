'use client';

import { useState, useTransition } from 'react';
import { Sparkles, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { generateAIPlanningAction } from '@/app/ai/actions';

type Site = {
  id: string;
  name: string;
  deadline: string | null;
};

type Props = {
  sites: Site[];
};

type PlanningResult = {
  orderedTasks: Array<{
    taskId: string;
    order: number;
    startDate: string;
    endDate: string;
    assignedWorkerId: string | null;
    priority: 'high' | 'medium' | 'low';
    taskTitle: string;
  }>;
  warnings: string[];
  reasoning: string;
};

export function AIPlanningSection({ sites }: Props) {
  const [selectedSiteId, setSelectedSiteId] = useState<string>('');
  const [isPending, startTransition] = useTransition();
  const [result, setResult] = useState<PlanningResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleGenerate() {
    if (!selectedSiteId) {
      setError('Sélectionnez un chantier');
      return;
    }

    setError(null);
    setResult(null);

    startTransition(async () => {
      try {
        const response = await generateAIPlanningAction(selectedSiteId);
        if (response.error) {
          setError(response.error);
        } else if (response.planning) {
          setResult(response.planning);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Erreur inconnue');
      }
    });
  }

  return (
    <div className="space-y-6">
      {/* Sélection du chantier */}
      <section className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
          Sélectionner un chantier
        </h2>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Choisissez le chantier pour lequel vous souhaitez générer un planning optimisé
        </p>
        <div className="mt-4">
          <select
            value={selectedSiteId}
            onChange={(e) => setSelectedSiteId(e.target.value)}
            className="w-full rounded-lg border border-zinc-200 bg-white px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-black/60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          >
            <option value="">-- Sélectionner un chantier --</option>
            {sites.map((site) => (
              <option key={site.id} value={site.id}>
                {site.name} {site.deadline ? `(${new Date(site.deadline).toLocaleDateString('fr-FR')})` : ''}
              </option>
            ))}
          </select>
        </div>
        <button
          type="button"
          onClick={handleGenerate}
          disabled={isPending || !selectedSiteId}
          className="mt-4 flex items-center gap-2 rounded-full bg-emerald-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-emerald-500 dark:hover:bg-emerald-600"
        >
          {isPending ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Génération en cours...
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              Générer le planning IA
            </>
          )}
        </button>
      </section>

      {/* Résultats */}
      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 dark:border-rose-900/60 dark:bg-rose-900/20">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-rose-600 dark:text-rose-400" />
            <div>
              <p className="text-sm font-semibold text-rose-900 dark:text-rose-100">
                Erreur
              </p>
              <p className="mt-1 text-sm text-rose-700 dark:text-rose-300">{error}</p>
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="space-y-4">
          {/* Raisonnement de l'IA */}
          {result.reasoning && (
            <section
              className={`rounded-2xl border p-6 ${
                result.reasoning.includes('algorithme de base') ||
                result.reasoning.includes('Erreur lors de l\'appel OpenAI')
                  ? 'border-amber-200 bg-amber-50 dark:border-amber-900/60 dark:bg-amber-900/20'
                  : 'border-emerald-200 bg-emerald-50 dark:border-emerald-900/60 dark:bg-emerald-900/20'
              }`}
            >
              <div className="flex items-start gap-3">
                <Sparkles
                  className={`h-5 w-5 ${
                    result.reasoning.includes('algorithme de base') ||
                    result.reasoning.includes('Erreur lors de l\'appel OpenAI')
                      ? 'text-amber-600 dark:text-amber-400'
                      : 'text-emerald-600 dark:text-emerald-400'
                  }`}
                />
                <div className="flex-1">
                  <h3
                    className={`text-sm font-semibold ${
                      result.reasoning.includes('algorithme de base') ||
                      result.reasoning.includes('Erreur lors de l\'appel OpenAI')
                        ? 'text-amber-900 dark:text-amber-100'
                        : 'text-emerald-900 dark:text-emerald-100'
                    }`}
                  >
                    {result.reasoning.includes('algorithme de base') ||
                    result.reasoning.includes('Erreur lors de l\'appel OpenAI')
                      ? '⚠️ Mode basique (OpenAI non disponible)'
                      : 'Analyse IA'}
                  </h3>
                  <p
                    className={`mt-2 text-sm ${
                      result.reasoning.includes('algorithme de base') ||
                      result.reasoning.includes('Erreur lors de l\'appel OpenAI')
                        ? 'text-amber-800 dark:text-amber-200'
                        : 'text-emerald-800 dark:text-emerald-200'
                    }`}
                  >
                    {result.reasoning}
                  </p>
                  {(result.reasoning.includes('algorithme de base') ||
                    result.reasoning.includes('Erreur lors de l\'appel OpenAI')) && (
                    <div className="mt-4 rounded-lg bg-white/50 p-3 text-xs dark:bg-black/20">
                      {result.reasoning.includes('429') || result.reasoning.includes('Quota') ? (
                        <>
                          <p className="font-semibold mb-2">⚠️ Quota OpenAI dépassé</p>
                          <p className="mb-2">Vous avez fait trop de requêtes. Solutions :</p>
                          <ol className="list-decimal list-inside space-y-1 mb-2">
                            <li>Attendez 1-2 minutes avant de réessayer</li>
                            <li>Vérifiez votre quota sur <a href="https://platform.openai.com/usage" target="_blank" rel="noopener noreferrer" className="underline">platform.openai.com/usage</a></li>
                            <li>Si nécessaire, passez à un plan payant OpenAI</li>
                          </ol>
                          <p className="text-xs opacity-75">En attendant, le planning est généré avec l'algorithme de base.</p>
                        </>
                      ) : (
                        <>
                          <p className="font-semibold mb-1">Pour activer l'IA OpenAI :</p>
                          <ol className="list-decimal list-inside space-y-1">
                            <li>Vérifiez que OPENAI_API_KEY est bien configurée dans Vercel</li>
                            <li>Redéployez votre application après avoir ajouté la variable</li>
                            <li>Vérifiez les logs Vercel pour voir les erreurs détaillées</li>
                          </ol>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </section>
          )}

          {/* Planning généré */}
          <section className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
                Planning généré ({result.orderedTasks.length} tâches)
              </h3>
            </div>

            <div className="mt-6 space-y-3">
              {result.orderedTasks.map((item) => (
                <div
                  key={item.taskId}
                  className="flex items-start gap-4 rounded-xl border border-zinc-100 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900"
                >
                  <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-emerald-100 text-sm font-semibold text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400">
                    {item.order}
                  </div>
                  <div className="flex-1">
                    <p className="font-semibold text-zinc-900 dark:text-white">
                      {item.taskTitle}
                    </p>
                    <div className="mt-2 flex flex-wrap items-center gap-4 text-xs text-zinc-500 dark:text-zinc-400">
                      <span>
                        📅 {new Date(item.startDate).toLocaleDateString('fr-FR')} →{' '}
                        {new Date(item.endDate).toLocaleDateString('fr-FR')}
                      </span>
                      <span
                        className={`rounded-full px-2 py-1 ${
                          item.priority === 'high'
                            ? 'bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-400'
                            : item.priority === 'medium'
                              ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400'
                              : 'bg-zinc-100 text-zinc-800 dark:bg-zinc-800 dark:text-zinc-400'
                        }`}
                      >
                        {item.priority === 'high' ? 'Priorité haute' : item.priority === 'medium' ? 'Priorité moyenne' : 'Priorité basse'}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Avertissements */}
          {result.warnings.length > 0 && (
            <section className="rounded-xl border border-amber-200 bg-amber-50 p-4 dark:border-amber-900/60 dark:bg-amber-900/20">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                <div>
                  <p className="text-sm font-semibold text-amber-900 dark:text-amber-100">
                    Avertissements
                  </p>
                  <ul className="mt-2 space-y-1">
                    {result.warnings.map((warning, index) => (
                      <li key={index} className="text-sm text-amber-800 dark:text-amber-200">
                        • {warning}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </section>
          )}
        </div>
      )}

      {/* Info */}
      <section className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          💡 <strong>Astuce :</strong> L'IA analyse vos tâches, identifie les dépendances, et
          génère un planning optimisé avec dates de début/fin et priorités. Les tâches sont
          classées par ordre logique d'exécution.
        </p>
      </section>
    </div>
  );
}

