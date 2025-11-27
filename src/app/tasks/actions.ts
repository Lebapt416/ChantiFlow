'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';
import { sendEmail } from '@/lib/email';
import { generateWorkerAccessCodeAlphanumeric } from '@/lib/worker-access';

export type AssignTaskState = {
  error?: string;
  success?: boolean;
};

export type ActionState = {
  error?: string;
  success?: boolean;
};

/**
 * Action pour ajouter une tâche
 */
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
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { error: 'Non authentifié.' };
  }

  // Vérifier que le chantier appartient à l'utilisateur
  const { data: site } = await supabase
    .from('sites')
    .select('id, created_by')
    .eq('id', siteId)
    .eq('created_by', user.id)
    .single();

  if (!site) {
    return { error: 'Chantier non trouvé ou accès refusé.' };
  }

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

  revalidatePath('/tasks');
  revalidatePath(`/site/${siteId}`);
  return { success: true };
}

/**
 * Action pour assigner un worker à une tâche
 */
export async function assignTaskAction(
  _prevState: AssignTaskState,
  formData: FormData,
): Promise<AssignTaskState> {
  const taskId = String(formData.get('taskId') ?? '');
  const workerId = String(formData.get('workerId') ?? '').trim();

  if (!taskId) {
    return { error: 'ID de tâche requis.' };
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { error: 'Non authentifié.' };
  }

  // Vérifier que la tâche appartient à un chantier de l'utilisateur
  const { data: task, error: taskError } = await supabase
    .from('tasks')
    .select('id, site_id')
    .eq('id', taskId)
    .single();

  if (taskError || !task) {
    return { error: 'Tâche non trouvée ou accès refusé.' };
  }

  // Vérifier que le chantier appartient à l'utilisateur
  const { data: site } = await supabase
    .from('sites')
    .select('id, created_by')
    .eq('id', task.site_id)
    .eq('created_by', user.id)
    .single();

  if (!site) {
    return { error: 'Accès refusé à cette tâche.' };
  }

  // Vérifier que le worker appartient à l'utilisateur et récupérer ses infos
  let worker: { id: string; name: string; email: string | null; access_code: string | null } | null = null;
  if (workerId) {
    const { data: workerData, error: workerError } = await supabase
      .from('workers')
      .select('id, name, email, created_by, access_code')
      .eq('id', workerId)
      .single();

    if (workerError || !workerData) {
      return { error: 'Worker non trouvé ou accès refusé.' };
    }

    // Vérifier que le worker appartient à l'utilisateur
    if (workerData.created_by !== user.id) {
      return { error: 'Vous ne pouvez assigner que vos propres workers.' };
    }

    worker = workerData;
  }

  // Mettre à jour la tâche avec l'assignation
  const updateData: { assigned_worker_id?: string | null } = {};
  if (workerId) {
    updateData.assigned_worker_id = workerId;
  } else {
    updateData.assigned_worker_id = null;
  }

  const { error } = await supabase
    .from('tasks')
    .update(updateData)
    .eq('id', taskId);

  if (error) {
    // Si la colonne n'existe pas, on essaie de la créer via une migration
    if (error.message.includes('assigned_worker_id') || error.message.includes('column')) {
      return { 
        error: 'La colonne assigned_worker_id n\'existe pas. Veuillez exécuter la migration SQL.' 
      };
    }
    return { error: error.message };
  }

  // Si un worker a été assigné, générer ou récupérer son code d'accès et envoyer un email
  if (worker && workerId) {
    const admin = createSupabaseAdminClient();
    
    // Récupérer les infos complètes de la tâche et du chantier
    const { data: taskDetails } = await admin
      .from('tasks')
      .select('title, sites!inner(name)')
      .eq('id', taskId)
      .single();

    // Générer ou récupérer le code d'accès
    let accessCode = worker.access_code;
    if (!accessCode) {
      // Générer un nouveau code au format 4 chiffres + 4 lettres
      accessCode = generateWorkerAccessCodeAlphanumeric();
      
      // Vérifier que le code respecte le format (4 chiffres + 4 lettres)
      if (!/^[0-9]{4}[A-Z]{4}$/.test(accessCode)) {
        console.error('Code généré invalide, régénération:', accessCode);
        accessCode = generateWorkerAccessCodeAlphanumeric();
      }
      
      // Vérifier que le code n'existe pas déjà (éviter les doublons)
      let attempts = 0;
      while (attempts < 10) {
        const { data: existingWorker } = await admin
          .from('workers')
          .select('id')
          .eq('access_code', accessCode)
          .maybeSingle();
        
        if (!existingWorker) {
          break; // Code unique trouvé
        }
        
        // Régénérer un nouveau code
        accessCode = generateWorkerAccessCodeAlphanumeric();
        attempts++;
      }
      
      // Mettre à jour le worker avec le code d'accès
      const { error: updateError } = await admin
        .from('workers')
        .update({ access_code: accessCode })
        .eq('id', workerId);
      
      if (updateError) {
        console.error('Erreur lors de la mise à jour du code d\'accès:', updateError);
      }
    }
    
    // S'assurer que le code est en majuscules et respecte le format
    accessCode = String(accessCode || '').toUpperCase().trim();
    
    // Vérifier le format final avant l'envoi (4 chiffres + 4 lettres)
    if (!/^[0-9]{4}[A-Z]{4}$/.test(accessCode)) {
      console.error('Code d\'accès invalide avant envoi email:', accessCode, 'Régénération...');
      // Régénérer si nécessaire
      accessCode = generateWorkerAccessCodeAlphanumeric();
      await admin
        .from('workers')
        .update({ access_code: accessCode })
        .eq('id', workerId);
    }
    
    // S'assurer que le code est en majuscules
    accessCode = accessCode.toUpperCase();

    // Envoyer l'email avec le code d'accès
    if (worker.email) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const siteName = (taskDetails as any)?.sites?.name || 'un chantier';
        const taskTitle = taskDetails?.title || 'une tâche';
        
        await sendEmail({
          to: worker.email,
          subject: `Nouvelle tâche assignée - ${taskTitle}`,
          html: `
            <!DOCTYPE html>
            <html>
              <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
              </head>
              <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
                  <h1 style="color: white; margin: 0; font-size: 24px;">ChantiFlow</h1>
                </div>
                <div style="background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px;">
                  <h2 style="color: #1f2937; margin-top: 0;">Bonjour ${worker.name || 'collaborateur'},</h2>
                  
                  <p style="color: #4b5563; font-size: 16px;">
                    Une nouvelle tâche vous a été assignée sur le chantier <strong>${siteName}</strong>.
                  </p>
                  
                  <div style="background: white; border-left: 4px solid #667eea; padding: 20px; margin: 20px 0; border-radius: 4px;">
                    <p style="margin: 0; font-size: 18px; font-weight: 600; color: #1f2937;">
                      ${taskTitle}
                    </p>
                  </div>
                  
                  <p style="color: #4b5563; font-size: 16px;">
                    Pour accéder à votre espace employé et consulter cette tâche, utilisez le code d'accès suivant :
                  </p>
                  
                  <div style="background: #1f2937; color: white; padding: 20px; text-align: center; border-radius: 8px; margin: 30px 0;">
                    <p style="margin: 0; font-size: 14px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">Code d'accès</p>
                    <p style="margin: 0; font-size: 32px; font-weight: bold; letter-spacing: 4px; font-family: 'Courier New', monospace; text-transform: uppercase;">
                      ${String(accessCode).toUpperCase().replace(/[^0-9A-Z]/g, '')}
                    </p>
                    <p style="margin: 10px 0 0 0; font-size: 12px; color: #9ca3af;">
                      Format: 4 chiffres + 4 lettres (ex: 1234ABCD)
                    </p>
                  </div>
                  
                  <div style="text-align: center; margin: 30px 0;">
                    <a href="${process.env.NEXT_PUBLIC_APP_BASE_URL || 'https://chantiflow.com'}/worker/login" 
                       style="display: inline-block; background: #667eea; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: 600; font-size: 16px;">
                      Accéder à mon espace
                    </a>
                  </div>
                  
                  <p style="color: #6b7280; font-size: 14px; margin-top: 30px; border-top: 1px solid #e5e7eb; padding-top: 20px;">
                    Ce code est unique et personnel. Ne le partagez pas.
                  </p>
                </div>
              </body>
            </html>
          `,
        });
      } catch (emailError) {
        console.error('Erreur lors de l\'envoi de l\'email:', emailError);
        // Ne pas bloquer l'assignation si l'email échoue
      }
    }
  }

  revalidatePath('/tasks');
  revalidatePath(`/site/${task.site_id}`);
  return { success: true };
}
