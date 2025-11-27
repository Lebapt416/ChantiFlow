'use server';

import { createSupabaseServerClient } from '@/lib/supabase/server';
import { generatePlanning } from '@/lib/ai/planning';

export type WorkerPlanningResult = {
  planning: Array<{
    taskId: string;
    taskTitle: string;
    order: number;
    startDate: string;
    endDate: string;
    assignedWorkerId: string | null;
    assignedWorkerIds?: string[];
    priority: 'high' | 'medium' | 'low';
    estimatedHours?: number;
  }>;
  error?: string;
};

export async function getWorkerPlanning(
  siteId: string,
  workerId: string,
): Promise<WorkerPlanningResult> {
  try {
    const supabase = await createSupabaseServerClient();

    // Charger les tâches et workers
    const [{ data: tasks }, { data: workers }, { data: site }] = await Promise.all([
      supabase
        .from('tasks')
        .select('id, title, required_role, duration_hours, status')
        .eq('site_id', siteId),
      supabase
        .from('workers')
        .select('id, name, email, role')
        .eq('site_id', siteId),
      supabase
        .from('sites')
        .select('deadline, postal_code')
        .eq('id', siteId)
        .single(),
    ]);

    if (!tasks || !workers) {
      return {
        planning: [],
        error: 'Impossible de charger les données du chantier.',
      };
    }

    // Générer le planning avec l'IA
    const pendingTasks = tasks.filter((t) => t.status === 'pending');
    
    console.log('📅 Génération planning:', {
      siteId,
      workerId,
      pendingTasksCount: pendingTasks.length,
      workersCount: workers?.length || 0,
    });

    if (pendingTasks.length === 0) {
      return {
        planning: [],
        error: 'Aucune tâche en attente à planifier.',
      };
    }

    const planningResult = await generatePlanning(
      pendingTasks,
      workers || [],
      site?.deadline || null,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (site as any)?.postal_code || undefined,
    );

    console.log('📅 Planning généré:', {
      orderedTasksCount: planningResult.orderedTasks.length,
      warnings: planningResult.warnings,
    });

    // Mapper les tâches avec leurs titres
    const allPlanning = planningResult.orderedTasks.map((p) => {
      const task = tasks.find((t) => t.id === p.taskId);
      // Gérer les deux formats: assignedWorkerIds (nouveau) ou assignedWorkerId (ancien)
      const assignedWorkerIds = 'assignedWorkerIds' in p && Array.isArray(p.assignedWorkerIds)
        ? p.assignedWorkerIds
        : p.assignedWorkerId
        ? [p.assignedWorkerId]
        : [];
      const assignedWorkerId = assignedWorkerIds.length > 0 ? assignedWorkerIds[0] : null;
      
      // Gérer estimatedHours (peut être présent ou non selon la version de generatePlanning)
      const estimatedHours = 'estimatedHours' in p && typeof p.estimatedHours === 'number'
        ? p.estimatedHours
        : undefined;
      
      return {
        taskId: p.taskId,
        taskTitle: task?.title || 'Tâche inconnue',
        order: p.order,
        startDate: p.startDate,
        endDate: p.endDate,
        assignedWorkerId,
        assignedWorkerIds,
        priority: p.priority,
        estimatedHours,
      };
    });

    // Filtrer pour ce worker OU afficher toutes les tâches si aucune n'est assignée
    const workerPlanning = allPlanning.filter(
      (p) => p.assignedWorkerId === workerId || !p.assignedWorkerId
    );

    console.log('📅 Planning filtré pour worker:', {
      workerId,
      totalPlanning: allPlanning.length,
      workerPlanningCount: workerPlanning.length,
      assignedTasks: allPlanning.filter((p) => p.assignedWorkerId === workerId).length,
    });

    return {
      planning: workerPlanning,
    };
  } catch (error) {
    console.error('Erreur chargement planning worker:', error);
    return {
      planning: [],
      error: error instanceof Error ? error.message : 'Erreur lors du chargement du planning.',
    };
  }
}

