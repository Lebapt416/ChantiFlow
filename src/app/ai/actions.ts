'use server';

import { createSupabaseServerClient } from '@/lib/supabase/server';
import { generateAIPlanning } from '@/lib/ai/openai-planning';

export type GenerateAIPlanningState = {
  error?: string;
  planning?: {
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
};

export async function generateAIPlanningAction(
  siteId: string,
): Promise<GenerateAIPlanningState & { workers?: Array<{ id: string; name: string; email: string; role: string | null }> }> {
  try {
    const supabase = await createSupabaseServerClient();

    // Récupérer les tâches et workers
    const [{ data: tasks }, { data: site }] = await Promise.all([
      supabase
        .from('tasks')
        .select('id, title, required_role, duration_hours, status, assigned_worker_id')
        .eq('site_id', siteId),
      supabase
        .from('sites')
        .select('deadline, name, created_by')
        .eq('id', siteId)
        .single(),
    ]);

    // Récupérer les workers du chantier
    const { data: siteWorkers } = await supabase
      .from('workers')
      .select('id, name, email, role')
      .eq('site_id', siteId);

    // Récupérer les workers réutilisables (sans site_id) qui sont assignés à des tâches de ce chantier
    const assignedWorkerIds = tasks
      ?.filter((task) => task.assigned_worker_id)
      .map((task) => task.assigned_worker_id)
      .filter((id): id is string => id !== null) || [];

    let accountWorkers: Array<{ id: string; name: string; email: string; role: string | null }> = [];
    
    if (site?.created_by && assignedWorkerIds.length > 0) {
      const { data: reusableWorkers } = await supabase
        .from('workers')
        .select('id, name, email, role')
        .eq('created_by', site.created_by)
        .is('site_id', null)
        .in('id', assignedWorkerIds);
      
      if (reusableWorkers) {
        accountWorkers = reusableWorkers;
      }
    }

    // Combiner les workers du chantier et les workers réutilisables assignés
    const allWorkers = [
      ...(siteWorkers || []),
      ...accountWorkers,
    ];

    // Dédupliquer par ID
    const workersMap = new Map<string, { id: string; name: string; email: string; role: string | null }>();
    allWorkers.forEach((worker) => {
      if (!workersMap.has(worker.id)) {
        workersMap.set(worker.id, worker);
      }
    });
    const workers = Array.from(workersMap.values());

    if (!tasks || !site) {
      return { error: 'Impossible de charger les données du chantier.' };
    }

    const pendingTasks = tasks.filter((task) => task.status === 'pending');

    if (pendingTasks.length === 0) {
      return { error: 'Aucune tâche en attente à planifier.' };
    }

    // Générer le planning avec l'IA OpenAI
    const planning = await generateAIPlanning(
      pendingTasks,
      workers,
      site.deadline,
      site.name,
      siteId,
    );

    // Mapper les tâches avec les workers assignés depuis la base de données
    const orderedTasksWithAssignments = planning.orderedTasks.map((task) => {
      const dbTask = pendingTasks.find((t) => t.id === task.taskId);
      const assignedWorkerIdFromDb = dbTask?.assigned_worker_id;
      
      // Si la tâche a un worker assigné en DB mais pas dans le planning IA, l'utiliser
      let finalAssignedWorkerIds = task.assignedWorkerIds || (task.assignedWorkerId ? [task.assignedWorkerId] : []);
      if (assignedWorkerIdFromDb && !finalAssignedWorkerIds.includes(assignedWorkerIdFromDb)) {
        finalAssignedWorkerIds = [assignedWorkerIdFromDb, ...finalAssignedWorkerIds];
      }
      
      return {
        ...task,
        taskTitle: dbTask?.title || 'Tâche inconnue',
        // Utiliser l'assignation de la DB si disponible, sinon celle de l'IA
        assignedWorkerId: assignedWorkerIdFromDb || task.assignedWorkerIds?.[0] || task.assignedWorkerId || null,
        assignedWorkerIds: finalAssignedWorkerIds.length > 0 ? finalAssignedWorkerIds : (assignedWorkerIdFromDb ? [assignedWorkerIdFromDb] : []),
        estimatedHours: task.estimatedHours || dbTask?.duration_hours || 8,
      };
    });

    return {
      planning: {
        orderedTasks: orderedTasksWithAssignments,
        warnings: planning.warnings,
        reasoning: planning.reasoning,
      },
      workers: workers,
    };
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : 'Erreur lors de la génération du planning',
    };
  }
}

