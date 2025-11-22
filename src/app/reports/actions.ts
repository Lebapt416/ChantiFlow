'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export type ValidateReportState = {
  error?: string;
  success?: boolean;
  message?: string;
};

export async function validateReportAction(
  _prevState: ValidateReportState,
  formData: FormData,
): Promise<ValidateReportState> {
  const reportId = String(formData.get('reportId') ?? '');
  const taskId = String(formData.get('taskId') ?? '');

  if (!reportId || !taskId) {
    return { error: 'Rapport et tâche requis.' };
  }
  try {
    const supabase = await createSupabaseServerClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      return { error: 'Non autorisé.' };
    }

    // Vérifier que la tâche appartient à un chantier de l'utilisateur
    const { data: task } = await supabase
      .from('tasks')
      .select('id, site_id, status')
      .eq('id', taskId)
      .single();

    if (!task) {
      return { error: 'Tâche non trouvée.' };
    }

    // Vérifier que le chantier appartient à l'utilisateur
    const { data: site } = await supabase
      .from('sites')
      .select('id, created_by')
      .eq('id', task.site_id)
      .single();

    if (!site || site.created_by !== user.id) {
      return { error: 'Non autorisé à valider ce rapport.' };
    }

    // Marquer la tâche comme terminée
    const { error: updateError } = await supabase
      .from('tasks')
      .update({ status: 'done' })
      .eq('id', taskId);

    if (updateError) {
      console.error('Erreur mise à jour tâche:', updateError);
      return { error: updateError.message };
    }

    revalidatePath('/reports');
    revalidatePath(`/report/${task.site_id}`);
    revalidatePath(`/site/${task.site_id}`);

    return {
      success: true,
      message: 'Rapport validé et tâche marquée comme terminée.',
    };
  } catch (error) {
    console.error('Erreur validation rapport:', error);
    return {
      error: error instanceof Error ? error.message : 'Erreur lors de la validation.',
    };
  }
}

