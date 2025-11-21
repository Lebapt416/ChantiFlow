'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { sendWorkerWelcomeEmail } from '@/lib/email';

export type ActionState = {
  error?: string;
  success?: boolean;
};

export async function addWorkerAction(
  _prevState: ActionState,
  formData: FormData,
): Promise<ActionState> {
  const name = String(formData.get('name') ?? '').trim();
  const email = String(formData.get('email') ?? '').trim();
  const role = String(formData.get('role') ?? '').trim();

  if (!name) {
    return { error: 'Nom requis.' };
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { error: 'Non authentifié.' };
  }

  // Vérifier si un worker avec le même email existe déjà pour ce compte
  if (email) {
    const { data: existingWorker } = await supabase
      .from('workers')
      .select('id')
      .eq('created_by', user.id)
      .is('site_id', null)
      .eq('email', email)
      .maybeSingle();

    if (existingWorker) {
      return { error: 'Un membre avec cet email existe déjà dans votre équipe.' };
    }
  }

  // Créer un worker au niveau du compte (sans site_id)
  const { error } = await supabase.from('workers').insert({
    created_by: user.id,
    name,
    email: email || null,
    role: role || null,
    site_id: null, // Worker au niveau du compte
  });

  if (error) {
    return { error: error.message };
  }

  // Envoyer un email de bienvenue si l'email est fourni
  if (email) {
    try {
      await sendWorkerWelcomeEmail({
        workerEmail: email,
        workerName: name,
        managerName: user.email || undefined,
      });
    } catch (error) {
      // Ne pas bloquer l'ajout si l'email échoue
      console.error('Erreur envoi email bienvenue:', error);
    }
  }

  revalidatePath('/team');
  return { success: true };
}

