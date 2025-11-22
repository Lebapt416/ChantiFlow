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
  // Les workers créés manuellement sont automatiquement approuvés
  const insertData: any = {
    created_by: user.id,
    name,
    email: email || null,
    role: role || null,
    site_id: null, // Worker au niveau du compte
  };

  // Essayer d'ajouter avec status 'approved' (workers créés manuellement sont approuvés)
  insertData.status = 'approved';
  let { error, data: insertedWorker } = await supabase
    .from('workers')
    .insert(insertData)
    .select('id')
    .single();

  // Si l'erreur est liée à la colonne status, réessayer sans
  if (error && (error.message.includes('status') || error.message.includes('column'))) {
    console.warn('Colonne status non trouvée, création sans status (sera considéré comme approuvé)');
    delete insertData.status;
    const { error: retryError, data: retryWorker } = await supabase
      .from('workers')
      .insert(insertData)
      .select('id')
      .single();
    
    if (retryError) {
      error = retryError;
    } else {
      error = null;
      insertedWorker = retryWorker;
    }
  }

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
  // Note: Pour les workers au niveau du compte, pas de code d'accès car ils ne sont pas encore assignés à un chantier
  // Le code sera généré et envoyé quand ils seront assignés à un chantier spécifique
  if (email) {
    try {
      console.log('📧 Tentative d\'envoi email de bienvenue (niveau compte) à:', email);
      const emailResult = await sendWorkerWelcomeEmail({
        workerEmail: email,
        workerName: name,
        managerName: user.email || undefined,
        // Pas de siteId ni accessCode car worker au niveau du compte
      });
      if (!emailResult.success) {
        console.warn('⚠️ Email non envoyé:', emailResult.error);
        // Ne pas retourner d'erreur, l'ajout du worker a réussi
      } else {
        console.log('✅ Email de bienvenue envoyé avec succès (niveau compte)');
      }
    } catch (error) {
      // Ne pas bloquer l'ajout si l'email échoue
      console.error('❌ Exception lors de l\'envoi email bienvenue:', error);
    }
  } else {
    console.log('ℹ️ Pas d\'email fourni, email de bienvenue non envoyé');
  }

  revalidatePath('/team');
  return { success: true };
}

export async function deleteWorkerAction(
  _prevState: ActionState,
  formData: FormData,
): Promise<ActionState> {
  const workerId = String(formData.get('workerId') ?? '').trim();

  if (!workerId) {
    return { error: 'ID worker requis.' };
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { error: 'Non authentifié.' };
  }

  // Vérifier que le worker appartient à l'utilisateur
  const { data: worker, error: fetchError } = await supabase
    .from('workers')
    .select('id, created_by, site_id')
    .eq('id', workerId)
    .single();

  if (fetchError || !worker) {
    return { error: 'Worker non trouvé.' };
  }

  // Vérifier les permissions : soit créé par l'utilisateur, soit dans un chantier de l'utilisateur
  if (worker.created_by && worker.created_by !== user.id) {
    // Vérifier si le worker est dans un chantier de l'utilisateur
    if (worker.site_id) {
      const { data: site } = await supabase
        .from('sites')
        .select('id, created_by')
        .eq('id', worker.site_id)
        .eq('created_by', user.id)
        .single();

      if (!site) {
        return { error: 'Vous n\'avez pas la permission de supprimer ce worker.' };
      }
    } else {
      return { error: 'Vous n\'avez pas la permission de supprimer ce worker.' };
    }
  } else if (worker.site_id) {
    // Worker lié à un chantier, vérifier que le chantier appartient à l'utilisateur
    const { data: site } = await supabase
      .from('sites')
      .select('id, created_by')
      .eq('id', worker.site_id)
      .eq('created_by', user.id)
      .single();

    if (!site) {
      return { error: 'Vous n\'avez pas la permission de supprimer ce worker.' };
    }
  }

  // Supprimer le worker
  const { error: deleteError } = await supabase
    .from('workers')
    .delete()
    .eq('id', workerId);

  if (deleteError) {
    return { error: `Erreur lors de la suppression: ${deleteError.message}` };
  }

  revalidatePath('/team');
  if (worker.site_id) {
    revalidatePath(`/site/${worker.site_id}`);
  }
  return { success: true };
}

