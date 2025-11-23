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
        .select('deadline, name')
        .eq('id', siteId)
        .single(),
    ]);

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
      workers || [],
      site.deadline,
      site.name,
      siteId,
    );

    return {
      planning: {
        orderedTasks: planning.orderedTasks.map((task) => ({
          ...task,
          taskTitle: pendingTasks.find((t) => t.id === task.taskId)?.title || 'Tâche inconnue',
          // Assurer la compatibilité avec l'ancien format
          assignedWorkerId: task.assignedWorkerIds?.[0] || task.assignedWorkerId || null,
          assignedWorkerIds: task.assignedWorkerIds || (task.assignedWorkerId ? [task.assignedWorkerId] : []),
          estimatedHours: task.estimatedHours || pendingTasks.find((t) => t.id === task.taskId)?.duration_hours || 8,
        })),
        warnings: planning.warnings,
        reasoning: planning.reasoning,
      },
      workers: workers || [],
    };
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : 'Erreur lors de la génération du planning',
    };
  }
}

