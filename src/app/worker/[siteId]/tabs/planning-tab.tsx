'use client';

import { useState, useEffect, useMemo } from 'react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { getWorkerPlanning } from '../actions';
import { CheckCircle2, XCircle, Calendar, Clock, AlertCircle, Eye } from 'lucide-react';
import { PlanningDetailModal } from './planning-detail-modal';
import { ModernPlanningView } from '@/components/modern-planning-view';

type PlanningTask = {
  taskId: string;
  taskTitle: string;
  order: number;
  startDate: string;
  endDate: string;
  assignedWorkerId: string | null;
  assignedWorkerIds?: string[];
  priority: 'high' | 'medium' | 'low';
  estimatedHours?: number;
  validated?: boolean;
  status?: string | null;
  requiredRole?: string | null;
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
  const [showDetailModal, setShowDetailModal] = useState(false);

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

  // Transformer les tâches en phases pour la vue moderne
  const modernPhases = useMemo(() => {
    if (planning.length === 0) return [];

    // Grouper par tâche (chaque tâche = une phase pour l'employé)
    return planning.map((task) => {
      let status: 'completed' | 'in_progress' | 'pending' = 'pending';
      let progress = 0;

      const taskStatus = task.status?.toLowerCase() || 'pending';
      
      if (task.validated || taskStatus === 'done' || taskStatus === 'completed') {
        status = 'completed';
        progress = 100;
      } else if (taskStatus === 'in_progress' || taskStatus === 'progress') {
        status = 'in_progress';
        // Calculer le pourcentage de progression basé sur les dates
        const now = new Date();
        const startDate = new Date(task.startDate);
        const endDate = new Date(task.endDate);
        const totalDuration = endDate.getTime() - startDate.getTime();
        if (totalDuration > 0) {
          const elapsed = Math.max(0, now.getTime() - startDate.getTime());
          progress = Math.min(Math.max(Math.round((elapsed / totalDuration) * 100), 10), 90);
        } else {
          progress = 50;
        }
      } else {
        // Vérifier si la tâche devrait être en cours (date actuelle entre startDate et endDate)
        const now = new Date();
        const startDate = new Date(task.startDate);
        const endDate = new Date(task.endDate);
        
        if (now >= startDate && now <= endDate) {
          status = 'in_progress';
          const totalDuration = endDate.getTime() - startDate.getTime();
          if (totalDuration > 0) {
            const elapsed = now.getTime() - startDate.getTime();
            progress = Math.min(Math.max(Math.round((elapsed / totalDuration) * 100), 10), 90);
          } else {
            progress = 50;
          }
        } else if (now > endDate) {
          status = 'in_progress';
          progress = 90;
        }
      }

      return {
        id: task.taskId,
        name: task.taskTitle,
        status,
        progress,
      };
    });
  }, [planning]);

  return (
    <div className="p-3 sm:p-4 md:p-6 space-y-4 sm:space-y-6">
      {planning.length === 0 ? (
        <div className="text-center py-12">
          <Calendar className="h-12 w-12 text-zinc-400 mx-auto mb-4" />
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Aucun planning généré pour le moment.
          </p>
          <p className="text-xs text-zinc-400 dark:text-zinc-500 mt-2">
            Le planning sera généré automatiquement par l&apos;IA.
          </p>
        </div>
      ) : (
        <>
          {/* Vue moderne du planning */}
          <ModernPlanningView
            siteName={siteName || 'Chantier'}
            phases={modernPhases}
            isAheadOfSchedule={false}
            showAIBadge={true}
          />

          {/* Bouton pour voir les détails */}
          <div className="flex justify-end">
            <button
              onClick={() => setShowDetailModal(true)}
              className="flex items-center gap-2 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-xs font-medium text-zinc-700 transition hover:bg-zinc-50 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
            >
              <Eye className="h-4 w-4" />
              <span>Voir en détail</span>
            </button>
          </div>
        </>
      )}

      {/* Modal de détails du planning */}
      {showDetailModal && (
        <PlanningDetailModal
          planning={planning}
          siteName={siteName || 'Chantier'}
          isOpen={showDetailModal}
          onClose={() => setShowDetailModal(false)}
        />
      )}
    </div>
  );
}

