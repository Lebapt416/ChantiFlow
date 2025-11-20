'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';

export type ChangePlanState = {
  error?: string;
  success?: boolean;
};

export async function changePlanAction(
  _prevState: ChangePlanState,
  formData: FormData,
): Promise<ChangePlanState> {
  const plan = String(formData.get('plan') ?? '').trim();

  if (!plan || !['basic', 'plus', 'pro'].includes(plan)) {
    return { error: 'Plan invalide.' };
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { error: 'Non authentifié.' };
  }

  // Utiliser l'admin client pour mettre à jour les métadonnées utilisateur
  const admin = createSupabaseAdminClient();

  // Mettre à jour les métadonnées utilisateur avec le plan
  const { error } = await admin.auth.admin.updateUserById(user.id, {
    user_metadata: {
      ...user.user_metadata,
      plan,
      plan_updated_at: new Date().toISOString(),
    },
  });

  if (error) {
    return { error: error.message };
  }

  revalidatePath('/account');
  revalidatePath('/dashboard');
  return { success: true };
}

