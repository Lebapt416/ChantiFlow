'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { capitalizeRoleWords } from '@/lib/utils/role-formatting';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';
import { sendWorkerWelcomeEmail, sendSiteCompletedEmail } from '@/lib/email';
import { generateAccessCode } from '@/lib/access-code';
import { canAddWorker } from '@/lib/plans';

export type ActionState = {
  error?: string;
  success?: boolean;
  message?: string;
};

type FeedbackPayload = {
  nombre_taches: number;
  complexite: number;
  duree_reelle: number;
};

export async function sendFeedbackToAI(payload: FeedbackPayload) {
  const baseUrl =
    process.env.NEXT_PUBLIC_PREDICTION_API_URL ||
    process.env.ML_API_URL ||
    '';

  if (!baseUrl) {
    console.warn('⚠️ Aucun endpoint ML configuré, feedback ignoré.');
    return;
  }

  const endpoint = `${baseUrl.replace(/\/$/, '')}/feedback`;

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      // Ne pas mettre en cache : on veut enregistrer chaque feedback
      cache: 'no-store',
    });

    if (!response.ok) {
      const detail = await response.text();
      console.warn('⚠️ Feedback AI rejeté:', response.status, detail);
    }
  } catch (error) {
    console.error('❌ Impossible d’envoyer le feedback IA:', error);
  }
}

export async function addTaskAction(
  _prevState: ActionState,
  formData: FormData,
): Promise<ActionState> {
  const siteId = String(formData.get('siteId') ?? '');
  const title = String(formData.get('title') ?? '').trim();
  const requiredRoleRaw = String(formData.get('required_role') ?? '').trim();
  const requiredRole = requiredRoleRaw ? capitalizeRoleWords(requiredRoleRaw) : null;
  const durationHours = Number(formData.get('duration_hours') ?? 0);

  if (!siteId || !title) {
    return { error: 'Site et titre requis.' };
  }

  const supabase = await createSupabaseServerClient();
  const { error } = await supabase.from('tasks').insert({
    site_id: siteId,
    title,
    required_role: requiredRole,
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
  const roleRaw = String(formData.get('role') ?? '').trim();
  const role = roleRaw ? capitalizeRoleWords(roleRaw) : null;

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

  const managerDisplayName =
    (typeof user.user_metadata?.full_name === 'string' && user.user_metadata.full_name.trim().length > 0
      ? user.user_metadata.full_name.trim()
      : undefined) || user.email || undefined;
  const managerEmail = user.email || undefined;

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

  // Vérifier les limites du plan pour les workers
  const { allowed: canAdd, reason: limitReason } = await canAddWorker(user.id);
  if (!canAdd) {
    return { error: limitReason || 'Limite d\'employés atteinte pour votre plan.' };
  }

  // Si un worker existant est sélectionné, le lier au chantier
  if (existingWorkerId) {
    // Vérifier que le worker appartient à l'utilisateur
    const { data: existingWorker } = await supabase
      .from('workers')
      .select('id, name, email, role, status')
      .eq('id', existingWorkerId)
      .eq('created_by', user.id)
      .is('site_id', null)
      .single();

    if (!existingWorker) {
      return { error: 'Worker non trouvé ou déjà assigné à un chantier.' };
    }

    // Si le worker était en attente, l'approuver automatiquement lors de l'assignation
    if (existingWorker.status === 'pending') {
      try {
        const { error: updateStatusError } = await supabase
          .from('workers')
          .update({ status: 'approved' })
          .eq('id', existingWorkerId);
        
        if (updateStatusError && !updateStatusError.message.includes('status') && !updateStatusError.message.includes('column')) {
          console.warn('⚠️ Erreur mise à jour statut worker:', updateStatusError.message);
        } else if (!updateStatusError) {
          console.log('✅ Statut worker mis à jour: pending → approved');
        }
      } catch {
        // Ne pas bloquer si la colonne status n'existe pas
        console.warn('⚠️ Impossible de mettre à jour le statut (colonne peut-être absente)');
      }
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
    let persistedAccessCode = accessCode;
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
        created_by: user.id,
      })
      .select('id, access_code')
      .single();

    if (error) {
      console.error('❌ Erreur insertion worker avec code:', error);
      // Si l'erreur est liée à access_code (colonne n'existe pas), continuer sans code
      if (error.message.includes('access_code') || error.message.includes('column')) {
        console.warn('⚠️ Colonne access_code non trouvée - migration SQL non exécutée');
        // Réessayer sans access_code
        const { error: retryError } = await supabase
          .from('workers')
          .insert({
            site_id: siteId,
            name: existingWorker.name,
            email: existingWorker.email,
            role: existingWorker.role,
            created_by: user.id,
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
      if (newWorker?.access_code) {
        persistedAccessCode = newWorker.access_code;
      }
      console.log('✅ Worker créé avec code:', persistedAccessCode);
    }

    // Envoyer un email de bienvenue si l'email est fourni
    if (existingWorker.email) {
      try {
        console.log('📧 Envoi email avec code d\'accès:', persistedAccessCode, 'type:', typeof persistedAccessCode);
        console.log('📧 Worker email:', existingWorker.email, 'Worker name:', existingWorker.name);
        const emailResult = await sendWorkerWelcomeEmail({
          workerEmail: existingWorker.email,
          workerName: existingWorker.name,
          siteName: site.name,
          siteId: siteId,
          managerName: managerDisplayName,
          managerEmail,
          accessCode: persistedAccessCode || undefined,
        });
        console.log('✅ Email envoyé avec succès, code:', persistedAccessCode, 'result:', emailResult);
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
    let persistedAccessCode = accessCode;
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
        created_by: user.id,
      })
      .select('id, access_code')
      .single();

    if (error) {
      console.error('❌ Erreur insertion worker avec code:', error);
      // Si l'erreur est liée à access_code (colonne n'existe pas), continuer sans code
      if (error.message.includes('access_code') || error.message.includes('column')) {
        console.warn('⚠️ Colonne access_code non trouvée - migration SQL non exécutée');
        // Réessayer sans access_code
        const { error: retryError } = await supabase
          .from('workers')
          .insert({
            site_id: siteId,
            name,
            email: email || null,
            role: role || null,
            created_by: user.id,
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
      if (newWorker?.access_code) {
        persistedAccessCode = newWorker.access_code;
      }
      console.log('✅ Worker créé avec code:', persistedAccessCode);
    }

    // Envoyer un email de bienvenue si l'email est fourni
    if (email) {
      try {
        console.log('📧 Envoi email avec code d\'accès:', persistedAccessCode, 'type:', typeof persistedAccessCode);
        console.log('📧 Worker email:', email, 'Worker name:', name);
        const emailResult = await sendWorkerWelcomeEmail({
          workerEmail: email,
          workerName: name,
          siteName: site.name,
          siteId: siteId,
          managerName: managerDisplayName,
          managerEmail,
          accessCode: persistedAccessCode || undefined,
        });
        console.log('✅ Email envoyé avec succès, code:', persistedAccessCode, 'result:', emailResult);
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
  const admin = createSupabaseAdminClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { error: 'Non authentifié.' };
  }

  // Vérifier que le chantier appartient à l'utilisateur
  const { data: site } = await supabase
    .from('sites')
    .select('id, name, created_by, created_at')
    .eq('id', siteId)
    .eq('created_by', user.id)
    .single();

  if (!site) {
    return { error: 'Chantier non trouvé ou accès refusé.' };
  }

  const { data: siteTasks } = await supabase
    .from('tasks')
    .select('id, duration_hours')
    .eq('site_id', siteId);

  const taskCount = siteTasks?.length ?? 0;
  const totalEstimatedHours = siteTasks?.reduce(
    (sum, task) => sum + (task.duration_hours ?? 8),
    0,
  ) ?? 0;
  const avgDurationHours = taskCount > 0 ? totalEstimatedHours / taskCount : 8;
  const derivedComplexity = Number(
    Math.min(10, Math.max(1, avgDurationHours / 4 + taskCount / 10)).toFixed(2),
  );

  // Récupérer tous les workers du chantier avant de les retirer
  const { data: workers } = await supabase
    .from('workers')
    .select('id, name, email, site_id')
    .eq('site_id', siteId);

  const completionDate = new Date();
  const completionISO = completionDate.toISOString();

  // Marquer le chantier comme terminé (ajouter un champ completed_at)
  const { error: siteError } = await supabase
    .from('sites')
    .update({
      completed_at: completionISO,
    })
    .eq('id', siteId);

  if (siteError) {
    console.error('❌ Erreur mise à jour chantier:', siteError);
    // Si l'erreur est liée à completed_at (colonne n'existe pas), continuer quand même
    if (
      siteError.message.includes('completed_at') ||
      siteError.message.includes('column')
    ) {
      console.warn('⚠️ Colonne completed_at non trouvée - migration SQL non exécutée');
      console.warn('⚠️ Le chantier sera terminé mais completed_at ne sera pas mis à jour');
      // On continue quand même pour retirer les workers et envoyer les emails
    } else {
      return { error: `Erreur lors de la finalisation du chantier: ${siteError.message}` };
    }
  } else {
    console.log('✅ Chantier marqué comme terminé (completed_at mis à jour)');
  }

  // Retirer tous les workers du chantier (mettre site_id à null)
  // On ne supprime pas les workers, on les retire juste du chantier
  const { error: workersError } = await admin
    .from('workers')
    .update({ site_id: null })
    .eq('site_id', siteId);

  if (workersError) {
    console.error('Erreur retrait workers:', workersError);
    // Ne pas bloquer si on ne peut pas retirer les workers, mais log l'erreur
  }

  // Récupérer l'email du créateur du chantier pour lui envoyer aussi un email
  let creatorEmail: string | undefined;
  try {
    const { data: creator } = await admin.auth.admin.getUserById(site.created_by);
    creatorEmail = creator?.user?.email;
  } catch (error) {
    console.warn('⚠️ Impossible de récupérer l\'email du créateur:', error);
    // Utiliser l'email de l'utilisateur actuel comme fallback
    creatorEmail = user.email || undefined;
  }

  // Envoyer un email à tous les workers qui ont un email + au créateur
  const emailRecipients: Array<{ email: string; name: string }> = [];

  // Ajouter les workers
  if (workers && workers.length > 0) {
    workers
      .filter((worker) => worker.email)
      .forEach((worker) => {
        emailRecipients.push({
          email: worker.email!,
          name: worker.name || 'Collaborateur',
        });
      });
  }

  // Ajouter le créateur du chantier s'il a un email et qu'il n'est pas déjà dans la liste
  if (creatorEmail && !emailRecipients.some((r) => r.email === creatorEmail)) {
    emailRecipients.push({
      email: creatorEmail,
      name: user.email || 'Chef de chantier',
    });
  }

  let emailMessage: string | undefined;
  let emailError: string | undefined;

  // Envoyer les emails
  if (emailRecipients.length > 0) {
    const emailPromises = emailRecipients.map((recipient) =>
      sendSiteCompletedEmail({
        workerEmail: recipient.email,
        workerName: recipient.name,
        siteName: site.name,
      }),
    );

    try {
      const results = await Promise.allSettled(emailPromises);
      const successCount = results.filter((r) => r.status === 'fulfilled').length;
      const failureCount = results.filter((r) => r.status === 'rejected').length;
      console.log(
        `✅ Emails de fin de chantier: ${successCount} envoyé(s), ${failureCount} échec(s)`,
      );

      if (failureCount > 0) {
        results.forEach((result, index) => {
          if (result.status === 'rejected') {
            console.error(`❌ Échec envoi email à ${emailRecipients[index].email}:`, result.reason);
          }
        });
      }

      if (successCount === 0) {
        emailError =
          "Impossible d'envoyer les emails de notification. Vérifie la configuration Resend.";
      } else {
        emailMessage = `Notification envoyée (${successCount}/${emailRecipients.length}).${
          failureCount > 0 ? ` ${failureCount} email(s) n'ont pas pu être envoyés.` : ''
        }`;
        if (failureCount > 0) {
          emailError = `${failureCount} email(s) n'ont pas pu être envoyés.`;
        }
      }
    } catch (error) {
      console.error('❌ Erreur envoi emails fin de chantier:', error);
      emailError = "Erreur lors de l'envoi des emails de notification.";
    }
  } else {
    emailMessage =
      'Aucun email disponible pour notifier ce chantier (aucun employé avec email).';
    console.warn("⚠️ Aucun destinataire d'email trouvé pour la notification de fin de chantier");
  }

  // Enregistrer le chantier comme terminé dans les métadonnées de l'utilisateur
  try {
    const { data: currentUser } = await admin.auth.admin.getUserById(user.id);
    const currentMetadata = currentUser.user?.user_metadata ?? {};
    const completedSitesMetadata = Array.isArray(currentMetadata.completedSites)
      ? [...currentMetadata.completedSites]
      : [];

    if (!completedSitesMetadata.includes(siteId)) {
      completedSitesMetadata.push(siteId);
      await admin.auth.admin.updateUserById(user.id, {
        user_metadata: {
          ...currentMetadata,
          completedSites: completedSitesMetadata,
        },
      });
      console.log('✅ Metadata utilisateur mis à jour avec le chantier terminé');
    }
  } catch (error) {
    console.warn('⚠️ Impossible de mettre à jour le metadata completedSites:', error);
  }

  const siteCreatedAtDate = site.created_at ? new Date(site.created_at) : null;
  const realDurationDays =
    siteCreatedAtDate && Number.isFinite(siteCreatedAtDate.getTime())
      ? (completionDate.getTime() - siteCreatedAtDate.getTime()) / (1000 * 60 * 60 * 24)
      : null;

  if (realDurationDays && Number.isFinite(realDurationDays)) {
    await sendFeedbackToAI({
      nombre_taches: Math.max(1, taskCount),
      complexite: derivedComplexity,
      duree_reelle: Number(Math.max(realDurationDays, 1).toFixed(2)),
    });
  }

  revalidatePath(`/site/${siteId}`);
  revalidatePath('/dashboard');
  revalidatePath('/sites');

  return {
    success: true,
    message: emailMessage ?? 'Chantier terminé. Aucune notification envoyée.',
    ...(emailError ? { error: emailError } : {}),
  };
}

