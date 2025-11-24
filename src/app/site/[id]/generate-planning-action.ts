'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { generatePlanning } from '@/lib/ai/planning';

export type GeneratePlanningState = {
  error?: string;
  success?: boolean;
  planning?: {
    orderedTasks: Array<{
      taskId: string;
      order: number;
      startDate: string;
      endDate: string;
      assignedWorkerId: string | null;
      priority: 'high' | 'medium' | 'low';
    }>;
    warnings: string[];
  };
};

export async function generatePlanningAction(
  siteId: string,
): Promise<GeneratePlanningState> {
  try {
    const supabase = await createSupabaseServerClient();

    // Récupérer les tâches et workers
    const [{ data: tasks }, { data: workers }] = await Promise.all([
      supabase
        .from('tasks')
        .select('id, title, required_role, duration_hours, status')
        .eq('site_id', siteId),
      supabase
        .from('workers')
        .select('id, name, email, role')
        .eq('site_id', siteId),
    ]);

    if (!tasks || !workers) {
      return { error: 'Impossible de charger les données du chantier.' };
    }

    // Récupérer la deadline et l'adresse du chantier
    const { data: site } = await supabase
      .from('sites')
      .select('deadline, address')
      .eq('id', siteId)
      .single();

    // Générer le planning avec l'IA (incluant optimisation météo)
    const planning = await generatePlanning(
      tasks,
      workers || [],
      site?.deadline || null,
      (site as any)?.address || undefined,
    );

    // Mettre à jour la planification sauvegardée (dernière génération uniquement)
    const plannedTaskIds = planning.orderedTasks.map((task) => task.taskId);

    // Réinitialiser les tâches non planifiées
    const unplannedIds = tasks
      .map((task) => task.id)
      .filter((taskId) => !plannedTaskIds.includes(taskId));

    if (unplannedIds.length > 0) {
      await supabase
        .from('tasks')
        .update({
          planned_start: null,
          planned_end: null,
          planned_order: null,
          planned_worker_id: null,
        })
        .in('id', unplannedIds);
    }

    // Appliquer les nouvelles dates
    for (const task of planning.orderedTasks) {
      await supabase
        .from('tasks')
        .update({
          planned_start: task.startDate,
          planned_end: task.endDate,
          planned_order: task.order,
          planned_worker_id: task.assignedWorkerId,
        })
        .eq('id', task.taskId);
    }

    revalidatePath(`/site/${siteId}`);

    return {
      success: true,
      planning: {
        orderedTasks: planning.orderedTasks,
        warnings: planning.warnings,
      },
    };
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : 'Erreur lors de la génération du planning',
    };
  }
}

