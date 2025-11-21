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
  // Essayer d'abord avec created_by (nouvelle structure)
  let insertData: any = {
    name,
    email: email || null,
    role: role || null,
    site_id: null, // Worker au niveau du compte
  };

  // Ajouter created_by seulement si la colonne existe (après migration)
  // On essaie d'abord avec created_by
  insertData.created_by = user.id;
  const { error } = await supabase.from('workers').insert(insertData);

  if (error) {
    // Si l'erreur est liée à created_by, essayer sans (fallback)
    if (error.message.includes('created_by') || error.message.includes('column') || error.code === '42703') {
      // Fallback : créer un worker lié au premier chantier de l'utilisateur
      const { data: firstSite } = await supabase
        .from('sites')
        .select('id')
        .eq('created_by', user.id)
        .limit(1)
        .single();

      if (firstSite) {
        const { error: fallbackError } = await supabase.from('workers').insert({
          site_id: firstSite.id,
          name,
          email: email || null,
          role: role || null,
        });

        if (fallbackError) {
          return { error: `Erreur lors de l'ajout. Veuillez exécuter la migration SQL dans Supabase (fichier migration-workers-to-account.sql). Détails: ${fallbackError.message}` };
        }
      } else {
        return { error: 'Veuillez d\'abord créer un chantier, ou exécutez la migration SQL pour activer les workers au niveau du compte. Voir le fichier migration-workers-to-account.sql' };
      }
    } else {
      return { error: error.message };
    }
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

