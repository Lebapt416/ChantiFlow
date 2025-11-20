'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export type ActionState = {
  error?: string;
  success?: boolean;
};

export async function addWorkerAction(
  _prevState: ActionState,
  formData: FormData,
): Promise<ActionState> {
  const siteId = String(formData.get('siteId') ?? '');
  const name = String(formData.get('name') ?? '').trim();
  const email = String(formData.get('email') ?? '').trim();
  const role = String(formData.get('role') ?? '').trim();

  if (!siteId || !name) {
    return { error: 'Chantier et nom requis.' };
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

  const { error } = await supabase.from('workers').insert({
    site_id: siteId,
    name,
    email: email || null,
    role: role || null,
  });

  if (error) {
    return { error: error.message };
  }

  revalidatePath('/team');
  revalidatePath(`/site/${siteId}`);
  return { success: true };
}

