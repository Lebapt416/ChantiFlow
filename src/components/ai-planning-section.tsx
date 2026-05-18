'use client';

import { useState, useTransition } from 'react';
import { Sparkles, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { generateAIPlanningAction } from '@/app/ai/actions';
import { InteractiveCalendar } from './interactive-calendar';

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
    assignedWorkerId?: string | null; // Ancien format (compatibilité)
    assignedWorkerIds?: string[]; // Nouveau format (collaboration)
    dependencies?: string[];
    priority: 'high' | 'medium' | 'low';
    estimatedHours?: number;
    taskTitle: string;
  }>;
  warnings: string[];
  reasoning: string;
};

type Worker = {
  id: string;
  name: string;
  email: string;
  role: string | null;
};

export function AIPlanningSection({ sites }: Props) {
  const [selectedSiteId, setSelectedSiteId] = useState<string>('');
  const [isPending, startTransition] = useTransition();
  const [result, setResult] = useState<PlanningResult | null>(null);
  const [workers, setWorkers] = useState<Worker[]>([]);
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
          setResult(null);
          setWorkers([]);
        } else if (response.planning) {
          setResult(response.planning);
          setWorkers(response.workers || []);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Erreur inconnue');
        setResult(null);
        setWorkers([]);
      }
    });
  }

  return (
    <div className="space-y-6">
      {/* Sélection du chantier */}
      <section className="rounded border border-rule-soft bg-paper p-6">
        <h2 className="text-lg font-semibold text-ink">
          Sélectionner un chantier
        </h2>
        <p className="mt-1 text-sm text-ink-3">
          Choisissez le chantier pour lequel vous souhaitez générer un planning optimisé
        </p>
        <div className="mt-4">
          <select
            value={selectedSiteId}
            onChange={(e) => setSelectedSiteId(e.target.value)}
            className="w-full rounded border border-rule-soft bg-paper px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-orange text-ink"
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
          className="mt-4 flex items-center gap-2 rounded bg-orange px-6 py-3 text-sm font-semibold text-paper transition hover:bg-orange-dark disabled:cursor-not-allowed disabled:opacity-50"
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
        <div className="rounded border border-rose-200 bg-rose-50 p-4 dark:border-rose-900/60 dark:bg-rose-900/20">
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
              className={`rounded border p-6 ${
                result.reasoning.includes('algorithme de base') ||
                result.reasoning.includes('Erreur lors de l\'appel Gemini') ||
                result.reasoning.includes('Gemini non disponible')
                  ? result.reasoning.includes('429') || result.reasoning.includes('Quota')
                    ? 'border-amber-200 bg-amber-50 dark:border-amber-900/60 dark:bg-amber-900/20'
                    : 'border-blue-200 bg-blue-50 dark:border-blue-900/60 dark:bg-blue-900/20'
                  : 'border-rule-soft bg-paper-2'
              }`}
            >
              <div className="flex items-start gap-3">
                <Sparkles
                  className={`h-5 w-5 ${
                    result.reasoning.includes('algorithme de base') ||
                    result.reasoning.includes('Erreur lors de l\'appel OpenAI')
                      ? result.reasoning.includes('429') || result.reasoning.includes('Quota')
                        ? 'text-amber-600 dark:text-amber-400'
                        : 'text-blue-600 dark:text-blue-400'
                      : 'text-green'
                  }`}
                />
                <div className="flex-1">
                  <h3
                    className={`text-sm font-semibold ${
                      result.reasoning.includes('algorithme de base') ||
                      result.reasoning.includes('Erreur lors de l\'appel OpenAI')
                        ? result.reasoning.includes('429') || result.reasoning.includes('Quota')
                          ? 'text-warn'
                          : 'text-blue'
                        : 'text-green'
                    }`}
                  >
                    {result.reasoning.includes('429') || result.reasoning.includes('Quota')
                      ? '⚠️ Quota Gemini dépassé'
                      : result.reasoning.includes('algorithme de base')
                        ? '⚠️ Mode basique'
                        : result.reasoning.includes('Gemini non disponible')
                          ? '🤖 IA Locale Avancée'
                          : 'Analyse IA'}
                  </h3>
                  <p
                    className={`mt-2 text-sm whitespace-pre-line ${
                      result.reasoning.includes('algorithme de base') ||
                      result.reasoning.includes('Erreur lors de l\'appel Gemini') ||
                      result.reasoning.includes('Gemini non disponible')
                        ? result.reasoning.includes('429') || result.reasoning.includes('Quota')
                          ? 'text-amber-800 dark:text-amber-200'
                          : 'text-blue-800 dark:text-blue-200'
                        : 'text-green'
                    }`}
                  >
                    {result.reasoning.includes('Gemini non disponible') && !result.reasoning.includes('429')
                      ? result.reasoning.split('). ')[1] || result.reasoning
                      : result.reasoning}
                  </p>
                  {(result.reasoning.includes('algorithme de base') ||
                    result.reasoning.includes('Erreur lors de l\'appel Gemini') ||
                    (result.reasoning.includes('Gemini non disponible') && result.reasoning.includes('429'))) && (
                    <div className="mt-4 rounded bg-paper p-3 text-xs">
                      {result.reasoning.includes('429') || result.reasoning.includes('Quota') ? (
                        <>
                          <p className="font-semibold mb-2">⚠️ Quota Google Gemini dépassé</p>
                          <p className="mb-2">Vous avez fait trop de requêtes. Solutions :</p>
                          <ol className="list-decimal list-inside space-y-1 mb-2">
                            <li>Attendez 1-2 minutes avant de réessayer</li>
                            <li>Vérifiez votre quota sur <a href="https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas" target="_blank" rel="noopener noreferrer" className="underline">Google Cloud Console</a></li>
                            <li>Si nécessaire, augmentez votre quota ou passez à un plan payant</li>
                          </ol>
                          <p className="text-xs opacity-75">En attendant, le planning est généré avec l&apos;algorithme de base.</p>
                        </>
                      ) : (
                        <>
                          <p className="font-semibold mb-1">Pour activer l&apos;IA Google Gemini :</p>
                          <ol className="list-decimal list-inside space-y-1">
                            <li>Vérifiez que GOOGLE_GEMINI_API_KEY est bien configurée dans Vercel</li>
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
          <section className="rounded border border-rule-soft bg-paper p-6">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green" />
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
                Planning généré ({result.orderedTasks.length} tâches)
              </h3>
            </div>

            <div className="mt-6 space-y-3">
              {result.orderedTasks.map((item) => (
                <div
                  key={item.taskId}
                  className="flex items-start gap-4 rounded border border-rule-soft bg-paper-2 p-4"
                >
                  <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded bg-paper text-sm font-semibold text-orange border border-orange">
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

          {/* Calendrier hebdomadaire interactif */}
          {result.orderedTasks.length > 0 && workers.length > 0 && (
            <InteractiveCalendar
              planning={result.orderedTasks.map((task) => ({
                ...task,
                taskTitle: task.taskTitle,
                // Utiliser les heures estimées de la tâche, limitées à 8h/jour
                hours: Math.min(task.estimatedHours || 8, 8),
              }))}
              workers={workers}
              taskDetails={{}}
              onUpdate={(taskId, workerId, day, hours) => {
                // Mise à jour via action serveur (à implémenter)
                console.log('Update:', { taskId, workerId, day, hours });
              }}
            />
          )}

          {/* Avertissements */}
          {result.warnings.length > 0 && (
            <section className="rounded border border-amber-200 bg-amber-50 p-4 dark:border-amber-900/60 dark:bg-amber-900/20">
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
      <section className="rounded border border-rule-soft bg-paper-2 p-4">
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          💡 <strong>Astuce :</strong> L&apos;IA analyse vos tâches, identifie les dépendances, et
          génère un planning optimisé avec dates de début/fin et priorités. Les tâches sont
          classées par ordre logique d&apos;exécution.
        </p>
        <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-500">
          ⚖️ <strong>Règles de travail appliquées :</strong> Maximum 8h de travail effectif par jour avec pause déjeuner obligatoire de 1h. 
          Les tâches dépassant 8h sont automatiquement réparties sur plusieurs jours.
        </p>
      </section>
    </div>
  );
}

