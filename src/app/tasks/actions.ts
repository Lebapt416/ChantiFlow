'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export type ActionState = {
  error?: string;
  success?: boolean;
};

export async function addTaskAction(
  _prevState: ActionState,
  formData: FormData,
): Promise<ActionState> {
  const siteId = String(formData.get('siteId') ?? '');
  const title = String(formData.get('title') ?? '').trim();
  const requiredRole = String(formData.get('required_role') ?? '').trim();
  const durationHours = Number(formData.get('duration_hours') ?? 0);

  if (!siteId || !title) {
    return { error: 'Chantier et titre requis.' };
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
    .select('id')
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

