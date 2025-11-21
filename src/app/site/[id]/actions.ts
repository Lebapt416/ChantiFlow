'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { sendWorkerWelcomeEmail, sendSiteCompletedEmail } from '@/lib/email';
import { generateAccessCode } from '@/lib/access-code';

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
    return { error: 'Site et titre requis.' };
  }

  const supabase = await createSupabaseServerClient();
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

  revalidatePath(`/site/${siteId}`);
  return { success: true };
}

export async function addWorkerAction(
  _prevState: ActionState,
  formData: FormData,
): Promise<ActionState> {
  const siteId = String(formData.get('siteId') ?? '');
  const existingWorkerId = String(formData.get('existingWorkerId') ?? '').trim();
  const name = String(formData.get('name') ?? '').trim();
  const email = String(formData.get('email') ?? '').trim();
  const role = String(formData.get('role') ?? '').trim();

  if (!siteId) {
    return { error: 'Site requis.' };
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { error: 'Non authentifié.' };
  }

  // Vérifier que le chantier appartient à l'utilisateur et récupérer ses infos
  const { data: site } = await supabase
    .from('sites')
    .select('id, name')
    .eq('id', siteId)
    .eq('created_by', user.id)
    .single();

  if (!site) {
    return { error: 'Chantier non trouvé ou accès refusé.' };
  }

  // Si un worker existant est sélectionné, le lier au chantier
  if (existingWorkerId) {
    // Vérifier que le worker appartient à l'utilisateur
    const { data: existingWorker } = await supabase
      .from('workers')
      .select('id, name, email, role')
      .eq('id', existingWorkerId)
      .eq('created_by', user.id)
      .is('site_id', null)
      .single();

    if (!existingWorker) {
      return { error: 'Worker non trouvé ou déjà assigné à un chantier.' };
    }

    // Vérifier si le worker n'est pas déjà assigné à ce chantier
    if (existingWorker.email) {
      const { data: alreadyAssigned } = await supabase
        .from('workers')
        .select('id')
        .eq('site_id', siteId)
        .eq('email', existingWorker.email)
        .maybeSingle();

      if (alreadyAssigned) {
        return { error: 'Ce membre est déjà assigné à ce chantier.' };
      }
    } else {
      // Si pas d'email, vérifier par nom
      const { data: alreadyAssigned } = await supabase
        .from('workers')
        .select('id')
        .eq('site_id', siteId)
        .eq('name', existingWorker.name)
        .maybeSingle();

      if (alreadyAssigned) {
        return { error: 'Ce membre est déjà assigné à ce chantier.' };
      }
    }

    // Générer un code d'accès unique
    let accessCode = generateAccessCode();
    let attempts = 0;
    let codeExists = true;
    
    // Vérifier que le code est unique (max 10 tentatives)
    while (codeExists && attempts < 10) {
      const { data: existing } = await supabase
        .from('workers')
        .select('id')
        .eq('access_code', accessCode)
        .maybeSingle();
      
      if (!existing) {
        codeExists = false;
      } else {
        accessCode = generateAccessCode();
        attempts++;
      }
    }

    console.log('🔑 Code d\'accès généré pour worker existant:', accessCode);

    // Créer une copie du worker pour ce chantier avec le code d'accès
    const { data: newWorker, error } = await supabase
      .from('workers')
      .insert({
        site_id: siteId,
        name: existingWorker.name,
        email: existingWorker.email,
        role: existingWorker.role,
        access_code: accessCode,
      })
      .select('id, access_code')
      .single();

    if (error) {
      console.error('❌ Erreur insertion worker avec code:', error);
      // Si l'erreur est liée à access_code (colonne n'existe pas), continuer sans code
      if (error.message.includes('access_code') || error.message.includes('column')) {
        console.warn('⚠️ Colonne access_code non trouvée - migration SQL non exécutée');
        // Réessayer sans access_code
        const { data: retryWorker, error: retryError } = await supabase
          .from('workers')
          .insert({
            site_id: siteId,
            name: existingWorker.name,
            email: existingWorker.email,
            role: existingWorker.role,
          })
          .select('id')
          .single();
        
        if (retryError) {
          return { error: `Erreur: ${retryError.message}. Veuillez exécuter la migration SQL (migration-worker-access-code.sql)` };
        }
        // Continuer sans code d'accès en base, mais on garde le code généré pour l'email
        // Le code sera affiché dans l'email même s'il n'est pas sauvegardé
        console.warn('⚠️ Code généré mais non sauvegardé (colonne manquante):', accessCode);
      } else {
        return { error: error.message };
      }
    } else {
      console.log('✅ Worker créé avec code:', newWorker?.access_code || accessCode);
    }

    // Envoyer un email de bienvenue si l'email est fourni
    if (existingWorker.email) {
      try {
        console.log('📧 Envoi email avec code d\'accès:', accessCode, 'type:', typeof accessCode);
        console.log('📧 Worker email:', existingWorker.email, 'Worker name:', existingWorker.name);
        const emailResult = await sendWorkerWelcomeEmail({
          workerEmail: existingWorker.email,
          workerName: existingWorker.name,
          siteName: site.name,
          siteId: siteId,
          managerName: user.email || undefined,
          accessCode: accessCode || undefined,
        });
        console.log('✅ Email envoyé avec succès, code:', accessCode, 'result:', emailResult);
      } catch (error) {
        // Ne pas bloquer l'ajout si l'email échoue
        console.error('❌ Erreur envoi email bienvenue:', error);
      }
    }
  } else {
    // Créer un nouveau worker directement lié au chantier
    if (!name) {
      return { error: 'Nom requis.' };
    }

    // Générer un code d'accès unique
    let accessCode = generateAccessCode();
    let attempts = 0;
    let codeExists = true;
    
    // Vérifier que le code est unique (max 10 tentatives)
    while (codeExists && attempts < 10) {
      const { data: existing } = await supabase
        .from('workers')
        .select('id')
        .eq('access_code', accessCode)
        .maybeSingle();
      
      if (!existing) {
        codeExists = false;
      } else {
        accessCode = generateAccessCode();
        attempts++;
      }
    }

    console.log('🔑 Code d\'accès généré pour nouveau worker:', accessCode);

    const { data: newWorker, error } = await supabase
      .from('workers')
      .insert({
        site_id: siteId,
        name,
        email: email || null,
        role: role || null,
        access_code: accessCode,
      })
      .select('id, access_code')
      .single();

    if (error) {
      console.error('❌ Erreur insertion worker avec code:', error);
      // Si l'erreur est liée à access_code (colonne n'existe pas), continuer sans code
      if (error.message.includes('access_code') || error.message.includes('column')) {
        console.warn('⚠️ Colonne access_code non trouvée - migration SQL non exécutée');
        // Réessayer sans access_code
        const { data: retryWorker, error: retryError } = await supabase
          .from('workers')
          .insert({
            site_id: siteId,
            name,
            email: email || null,
            role: role || null,
          })
          .select('id')
          .single();
        
        if (retryError) {
          return { error: `Erreur: ${retryError.message}. Veuillez exécuter la migration SQL (migration-worker-access-code.sql)` };
        }
        // Continuer sans code d'accès en base, mais on garde le code généré pour l'email
        // Le code sera affiché dans l'email même s'il n'est pas sauvegardé
        console.warn('⚠️ Code généré mais non sauvegardé (colonne manquante):', accessCode);
      } else {
        return { error: error.message };
      }
    } else {
      console.log('✅ Worker créé avec code:', newWorker?.access_code || accessCode);
    }

    // Envoyer un email de bienvenue si l'email est fourni
    if (email) {
      try {
        console.log('📧 Envoi email avec code d\'accès:', accessCode, 'type:', typeof accessCode);
        console.log('📧 Worker email:', email, 'Worker name:', name);
        const emailResult = await sendWorkerWelcomeEmail({
          workerEmail: email,
          workerName: name,
          siteName: site.name,
          siteId: siteId,
          managerName: user.email || undefined,
          accessCode: accessCode || undefined,
        });
        console.log('✅ Email envoyé avec succès, code:', accessCode, 'result:', emailResult);
      } catch (error) {
        // Ne pas bloquer l'ajout si l'email échoue
        console.error('❌ Erreur envoi email bienvenue:', error);
      }
    }
  }

  revalidatePath(`/site/${siteId}`);
  return { success: true };
}

export async function completeTaskAction(formData: FormData) {
  const siteId = String(formData.get('siteId') ?? '');
  const taskId = String(formData.get('taskId') ?? '');

  if (!siteId || !taskId) {
    return;
  }

  const supabase = await createSupabaseServerClient();
  await supabase
    .from('tasks')
    .update({ status: 'done' })
    .eq('id', taskId);

  revalidatePath(`/site/${siteId}`);
}

export async function completeSiteAction(
  _prevState: ActionState,
  formData: FormData,
): Promise<ActionState> {
  const siteId = String(formData.get('siteId') ?? '');

  if (!siteId) {
    return { error: 'Site requis.' };
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
    .select('id, name, created_by')
    .eq('id', siteId)
    .eq('created_by', user.id)
    .single();

  if (!site) {
    return { error: 'Chantier non trouvé ou accès refusé.' };
  }

  // Récupérer tous les workers du chantier avant de les retirer
  const { data: workers } = await supabase
    .from('workers')
    .select('id, name, email, site_id')
    .eq('site_id', siteId);

  // Marquer le chantier comme terminé (ajouter un champ completed_at)
  const { error: siteError } = await supabase
    .from('sites')
    .update({ 
      completed_at: new Date().toISOString(),
    })
    .eq('id', siteId);

  if (siteError) {
    console.error('Erreur mise à jour chantier:', siteError);
    return { error: `Erreur lors de la finalisation du chantier: ${siteError.message}` };
  }

  // Retirer tous les workers du chantier (mettre site_id à null)
  // On ne supprime pas les workers, on les retire juste du chantier
  const { error: workersError } = await supabase
    .from('workers')
    .update({ site_id: null })
    .eq('site_id', siteId);

  if (workersError) {
    console.error('Erreur retrait workers:', workersError);
    // Ne pas bloquer si on ne peut pas retirer les workers, mais log l'erreur
  }

  // Envoyer un email à tous les workers qui ont un email
  if (workers && workers.length > 0) {
    const emailPromises = workers
      .filter((worker) => worker.email)
      .map((worker) =>
        sendSiteCompletedEmail({
          workerEmail: worker.email!,
          workerName: worker.name || 'Collaborateur',
          siteName: site.name,
        })
      );

    try {
      await Promise.allSettled(emailPromises);
      console.log(`✅ Emails de fin de chantier envoyés à ${emailPromises.length} employé(s)`);
    } catch (error) {
      console.error('Erreur envoi emails fin de chantier:', error);
      // Ne pas bloquer si l'envoi d'email échoue
    }
  }

  revalidatePath(`/site/${siteId}`);
  revalidatePath('/dashboard');
  revalidatePath('/sites');
  return { success: true };
}

