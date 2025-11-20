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

    // Récupérer la deadline du chantier
    const { data: site } = await supabase
      .from('sites')
      .select('deadline')
      .eq('id', siteId)
      .single();

    // Générer le planning avec l'IA
    const planning = await generatePlanning(tasks, workers || [], site?.deadline || null);

    // Mettre à jour l'ordre des tâches dans la base de données
    // On pourrait créer une table "planning" ou ajouter un champ "order" aux tâches
    // Pour l'instant, on retourne juste le planning généré

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

