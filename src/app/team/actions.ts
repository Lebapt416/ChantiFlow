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
    try {
      const { data: existingWorker, error: checkError } = await supabase
        .from('workers')
        .select('id')
        .eq('created_by', user.id)
        .is('site_id', null)
        .eq('email', email)
        .maybeSingle();

      // Si l'erreur est liée à created_by, on ignore (migration non exécutée)
      if (checkError && !checkError.message.includes('created_by') && !checkError.message.includes('column') && checkError.code !== '42703') {
        console.warn('Erreur vérification worker existant:', checkError);
      }

      if (existingWorker) {
        return { error: 'Un membre avec cet email existe déjà dans votre équipe.' };
      }
    } catch (checkError: any) {
      // Si la colonne created_by n'existe pas encore, on continue quand même
      console.warn('Erreur vérification worker existant:', checkError?.message);
    }
  }

  // Créer un worker au niveau du compte (sans site_id)
  const insertData = {
    created_by: user.id,
    name,
    email: email || null,
    role: role || null,
    site_id: null, // Worker au niveau du compte
  };

  const { error, data: insertedWorker } = await supabase
    .from('workers')
    .insert(insertData)
    .select('id')
    .single();

  if (error) {
    // Vérifier le type d'erreur
    const errorMessage = error.message || '';
    const errorCode = error.code || '';
    
    // Erreur de politique RLS
    if (errorMessage.includes('policy') || errorMessage.includes('permission') || errorCode === '42501') {
      return { 
        error: `Erreur de permissions. Vérifiez que les politiques RLS sont correctement configurées. Détails: ${errorMessage}. Exécutez la migration SQL si ce n'est pas déjà fait.` 
      };
    }
    
    // Erreur de contrainte unique
    if (errorMessage.includes('unique') || errorMessage.includes('duplicate') || errorCode === '23505') {
      return { error: 'Un membre avec cet email existe déjà dans votre équipe.' };
    }
    
    // Autre erreur
    return { 
      error: `Erreur lors de l'ajout: ${errorMessage} (Code: ${errorCode}). Vérifiez les logs pour plus de détails.` 
    };
  }

  if (!insertedWorker) {
    return { error: 'Le worker a été créé mais aucune donnée n\'a été retournée.' };
  }

  // Envoyer un email de bienvenue si l'email est fourni (ne bloque pas si ça échoue)
  if (email) {
    try {
      const emailResult = await sendWorkerWelcomeEmail({
        workerEmail: email,
        workerName: name,
        managerName: user.email || undefined,
      });
      if (!emailResult.success) {
        console.warn('Email non envoyé:', emailResult.error);
      }
    } catch (error) {
      // Ne pas bloquer l'ajout si l'email échoue
      console.error('Erreur envoi email bienvenue:', error);
    }
  }

  revalidatePath('/team');
  return { success: true };
}

