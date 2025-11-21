'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { sendWorkerWelcomeEmail } from '@/lib/email';
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
        // Continuer sans code d'accès
        accessCode = undefined;
      } else {
        return { error: error.message };
      }
    } else {
      console.log('✅ Worker créé avec code:', newWorker?.access_code || accessCode);
    }

    // Envoyer un email de bienvenue si l'email est fourni
    if (existingWorker.email) {
      try {
        console.log('📧 Envoi email avec code d\'accès:', accessCode);
        await sendWorkerWelcomeEmail({
          workerEmail: existingWorker.email,
          workerName: existingWorker.name,
          siteName: site.name,
          siteId: siteId,
          managerName: user.email || undefined,
          accessCode: accessCode,
        });
        console.log('✅ Email envoyé avec succès, code:', accessCode);
      } catch (error) {
        // Ne pas bloquer l'ajout si l'email échoue
        console.error('Erreur envoi email bienvenue:', error);
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
        // Continuer sans code d'accès
        accessCode = undefined;
      } else {
        return { error: error.message };
      }
    } else {
      console.log('✅ Worker créé avec code:', newWorker?.access_code || accessCode);
    }

    // Envoyer un email de bienvenue si l'email est fourni
    if (email) {
      try {
        console.log('📧 Envoi email avec code d\'accès:', accessCode);
        await sendWorkerWelcomeEmail({
          workerEmail: email,
          workerName: name,
          siteName: site.name,
          siteId: siteId,
          managerName: user.email || undefined,
          accessCode: accessCode,
        });
        console.log('✅ Email envoyé avec succès, code:', accessCode);
      } catch (error) {
        // Ne pas bloquer l'ajout si l'email échoue
        console.error('Erreur envoi email bienvenue:', error);
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

