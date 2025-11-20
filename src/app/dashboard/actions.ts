'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export type CreateSiteState = {
  error?: string;
  success?: boolean;
};

export async function createSiteAction(
  _prevState: CreateSiteState,
  formData: FormData,
): Promise<CreateSiteState> {
  const name = String(formData.get('name') ?? '').trim();
  const deadline = String(formData.get('deadline') ?? '');

  if (!name || !deadline) {
    return { error: 'Nom et deadline sont requis.' };
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();

  if (userError || !user) {
    return { error: 'Session expirée, reconnecte-toi.' };
  }

  const { error } = await supabase.from('sites').insert({
    name,
    deadline,
    created_by: user.id,
  });

  if (error) {
    return { error: error.message };
  }

  revalidatePath('/dashboard');
  return { success: true };
}

