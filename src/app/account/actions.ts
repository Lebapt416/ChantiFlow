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

  // Si c'est le plan Basic, on peut le changer directement (gratuit)
  if (plan === 'basic') {
    const admin = createSupabaseAdminClient();
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

  // Pour les plans payants (Plus/Pro), vérifier si c'est l'admin
  const { isAdminUser } = await import('@/lib/stripe');
  if (isAdminUser(user.email)) {
    // Admin : changement gratuit sans paiement
    const admin = createSupabaseAdminClient();
    const { error } = await admin.auth.admin.updateUserById(user.id, {
      user_metadata: {
        ...user.user_metadata,
        plan,
        plan_updated_at: new Date().toISOString(),
        admin_free_plan: true,
      },
    });

    if (error) {
      return { error: error.message };
    }

    revalidatePath('/account');
    revalidatePath('/dashboard');
    return { success: true };
  }

  // Pour les autres utilisateurs, rediriger vers Stripe checkout
  // Cette action ne sera pas appelée directement, mais via l'API checkout
  return { error: 'Redirection vers le paiement requise.' };
}

