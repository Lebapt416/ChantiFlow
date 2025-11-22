'use client';

import { useState, useEffect } from 'react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { getWorkerPlanning } from '../actions';
import { CheckCircle2, XCircle, Calendar, Clock, AlertCircle } from 'lucide-react';

type PlanningTask = {
  taskId: string;
  taskTitle: string;
  order: number;
  startDate: string;
  endDate: string;
  assignedWorkerId: string | null;
  priority: 'high' | 'medium' | 'low';
  validated?: boolean;
};

type Props = {
  siteId: string;
  workerId: string;
};

export function PlanningTab({ siteId, workerId }: Props) {
  const [planning, setPlanning] = useState<PlanningTask[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [siteName, setSiteName] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadPlanning() {
      const supabase = createSupabaseBrowserClient();

      try {
        // Charger le nom du chantier
        const { data: site } = await supabase
          .from('sites')
          .select('name')
          .eq('id', siteId)
          .single();

        if (site) {
          setSiteName(site.name);
        }

        // Récupérer le planning via l'action serveur
        console.log('📅 Chargement planning pour:', { siteId, workerId });
        const result = await getWorkerPlanning(siteId, workerId);

        console.log('📅 Résultat planning:', {
          planningCount: result.planning.length,
          error: result.error,
          planning: result.planning,
        });

        if (result.error) {
          setError(result.error);
          setIsLoading(false);
          return;
        }

        // Ajouter le champ validated
        const planningWithValidation = result.planning.map((task) => ({
          ...task,
          validated: false, // Par défaut non validé
        }));

        console.log('📅 Planning avec validation:', planningWithValidation);
        setPlanning(planningWithValidation);
      } catch (err) {
        console.error('Erreur chargement planning:', err);
        setError('Erreur lors du chargement du planning.');
      } finally {
        setIsLoading(false);
      }
    }

    loadPlanning();
  }, [siteId, workerId]);

  const handleValidate = async (taskId: string, validated: boolean) => {
    // Mettre à jour l'état local
    setPlanning((prev) =>
      prev.map((task) =>
        task.taskId === taskId ? { ...task, validated } : task,
      ),
    );

    // Ici, on pourrait sauvegarder la validation dans la base de données
    // Pour l'instant, on garde juste l'état local
    console.log('Validation planning:', { taskId, validated });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-zinc-300 border-t-emerald-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900/60 dark:bg-rose-900/20 dark:text-rose-200">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5" />
            {error}
          </div>
        </div>
      </div>
    );
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('fr-FR', {
      day: 'numeric',
      month: 'short',
    });
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'border-rose-200 bg-rose-50 text-rose-900 dark:border-rose-900/60 dark:bg-rose-900/20 dark:text-rose-200';
      case 'medium':
        return 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900/60 dark:bg-amber-900/20 dark:text-amber-200';
      default:
        return 'border-zinc-200 bg-zinc-50 text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300';
    }
  };

  return (
    <div className="p-3 sm:p-4 md:p-6 space-y-4 sm:space-y-6">
      <div>
        <h2 className="text-base sm:text-lg font-semibold text-zinc-900 dark:text-white mb-1">
          Mon emploi du temps
        </h2>
        <p className="text-xs sm:text-sm text-zinc-600 dark:text-zinc-400">
          {siteName || 'Chantier'} • {planning.length} tâche{planning.length > 1 ? 's' : ''} planifiée{planning.length > 1 ? 's' : ''}
        </p>
      </div>

      {planning.length === 0 ? (
        <div className="text-center py-12">
          <Calendar className="h-12 w-12 text-zinc-400 mx-auto mb-4" />
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Aucun planning généré pour le moment.
          </p>
          <p className="text-xs text-zinc-400 dark:text-zinc-500 mt-2">
            Le planning sera généré automatiquement par l'IA.
          </p>
        </div>
      ) : (
        <div className="space-y-2 sm:space-y-3">
          {planning.map((task) => (
            <div
              key={task.taskId}
              className={`rounded-lg border p-3 sm:p-4 transition ${
                task.validated
                  ? 'border-emerald-200 bg-emerald-50 dark:border-emerald-900/60 dark:bg-emerald-900/20'
                  : getPriorityColor(task.priority)
              }`}
            >
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="font-semibold text-sm sm:text-base text-zinc-900 dark:text-white break-words">
                      {task.taskTitle}
                    </h3>
                    {task.validated && (
                      <CheckCircle2 className="h-4 w-4 sm:h-5 sm:w-5 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
                    )}
                  </div>
                  
                  <div className="flex flex-wrap items-center gap-2 sm:gap-3 text-xs text-zinc-600 dark:text-zinc-400">
                    <div className="flex items-center gap-1">
                      <Calendar className="h-3 w-3" />
                      <span>
                        {formatDate(task.startDate)} - {formatDate(task.endDate)}
                      </span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      <span>Ordre: {task.order}</span>
                    </div>
                    <span className="px-2 py-0.5 rounded-full bg-zinc-200 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 text-[10px] sm:text-xs">
                      {task.priority === 'high' ? 'Priorité haute' : task.priority === 'medium' ? 'Priorité moyenne' : 'Priorité basse'}
                    </span>
                  </div>
                </div>

                <div className="flex sm:flex-col gap-2">
                  {!task.validated ? (
                    <button
                      onClick={() => handleValidate(task.taskId, true)}
                      className="flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium text-white transition hover:bg-emerald-700 active:scale-95 w-full sm:w-auto"
                    >
                      <CheckCircle2 className="h-3 w-3 sm:h-4 sm:w-4" />
                      <span className="sm:inline">Valider</span>
                    </button>
                  ) : (
                    <button
                      onClick={() => handleValidate(task.taskId, false)}
                      className="flex items-center justify-center gap-2 rounded-lg border border-zinc-300 bg-white px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium text-zinc-700 transition hover:bg-zinc-50 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700 active:scale-95 w-full sm:w-auto"
                    >
                      <XCircle className="h-3 w-3 sm:h-4 sm:w-4" />
                      <span className="sm:inline">Annuler</span>
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

