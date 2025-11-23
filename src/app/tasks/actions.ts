'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export type AssignTaskState = {
  error?: string;
  success?: boolean;
};

export type ActionState = {
  error?: string;
  success?: boolean;
};

/**
 * Action pour ajouter une tâche
 */
export async function addTaskAction(
  _prevState: ActionState,
  formData: FormData,
): Promise<ActionState> {
  const siteId = String(formData.get('siteId') ?? '');
  const title = String(formData.get('title') ?? '').trim();
  const requiredRole = String(formData.get('required_role') ?? '').trim();
  const durationHours = Number(formData.get('duration_hours') ?? 0);

  if (!siteId || !title) {
    return { error: 'Site et titre requis.' };
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { error: 'Non authentifié.' };
  }

  // Vérifier que le chantier appartient à l'utilisateur
  const { data: site } = await supabase
    .from('sites')
    .select('id, created_by')
    .eq('id', siteId)
    .eq('created_by', user.id)
    .single();

  if (!site) {
    return { error: 'Chantier non trouvé ou accès refusé.' };
  }

  const { error } = await supabase.from('tasks').insert({
    site_id: siteId,
    title,
    required_role: requiredRole || null,
    duration_hours: Number.isFinite(durationHours) ? durationHours : null,
    status: 'pending',
  });

  if (error) {
    return { error: error.message };
  }

  revalidatePath('/tasks');
  revalidatePath(`/site/${siteId}`);
  return { success: true };
}

/**
 * Action pour assigner un worker à une tâche
 */
export async function assignTaskAction(
  _prevState: AssignTaskState,
  formData: FormData,
): Promise<AssignTaskState> {
  const taskId = String(formData.get('taskId') ?? '');
  const workerId = String(formData.get('workerId') ?? '').trim();

  if (!taskId) {
    return { error: 'ID de tâche requis.' };
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { error: 'Non authentifié.' };
  }

  // Vérifier que la tâche appartient à un chantier de l'utilisateur
  const { data: task, error: taskError } = await supabase
    .from('tasks')
    .select('id, site_id')
    .eq('id', taskId)
    .single();

  if (taskError || !task) {
    return { error: 'Tâche non trouvée ou accès refusé.' };
  }

  // Vérifier que le chantier appartient à l'utilisateur
  const { data: site } = await supabase
    .from('sites')
    .select('id, created_by')
    .eq('id', task.site_id)
    .eq('created_by', user.id)
    .single();

  if (!site) {
    return { error: 'Accès refusé à cette tâche.' };
  }

  // Vérifier que le worker appartient à l'utilisateur
  if (workerId) {
    const { data: worker, error: workerError } = await supabase
      .from('workers')
      .select('id, created_by')
      .eq('id', workerId)
      .single();

    if (workerError || !worker) {
      return { error: 'Worker non trouvé ou accès refusé.' };
    }

    // Vérifier que le worker appartient à l'utilisateur
    if (worker.created_by !== user.id) {
      return { error: 'Vous ne pouvez assigner que vos propres workers.' };
    }
  }

  // Mettre à jour la tâche avec l'assignation
  const updateData: { assigned_worker_id?: string | null } = {};
  if (workerId) {
    updateData.assigned_worker_id = workerId;
  } else {
    updateData.assigned_worker_id = null;
  }

  const { error } = await supabase
    .from('tasks')
    .update(updateData)
    .eq('id', taskId);

  if (error) {
    // Si la colonne n'existe pas, on essaie de la créer via une migration
    if (error.message.includes('assigned_worker_id') || error.message.includes('column')) {
      return { 
        error: 'La colonne assigned_worker_id n\'existe pas. Veuillez exécuter la migration SQL.' 
      };
    }
    return { error: error.message };
  }

  revalidatePath('/tasks');
  revalidatePath(`/site/${task.site_id}`);
  return { success: true };
}
