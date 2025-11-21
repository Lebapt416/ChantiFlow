'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';
import { sendReportNotificationEmail } from '@/lib/email';

export type ReportState = {
  error?: string;
  success?: boolean;
};

const REPORT_BUCKET = 'reports';

export async function submitReportAction(
  _prevState: ReportState,
  formData: FormData,
): Promise<ReportState> {
  const siteIdParam = String(formData.get('siteId') ?? '');
  const taskId = String(formData.get('taskId') ?? '');
  const email = String(formData.get('email') ?? '').trim().toLowerCase();
  const name = String(formData.get('name') ?? '').trim();
  const role = String(formData.get('role') ?? '').trim();
  const description = String(formData.get('description') ?? '').trim();
  const markDone = formData.get('mark_done') === 'on';
  const file = formData.get('photo') as File | null;

  if (!siteIdParam || !taskId || !email) {
    return { error: 'Site, tâche et email sont requis.' };
  }

  const siteId = siteIdParam;

  const admin = createSupabaseAdminClient();

  // Récupérer les infos du chantier pour trouver le manager
  const { data: site } = await admin
    .from('sites')
    .select('id, created_by')
    .eq('id', siteId)
    .single();

  if (!site) {
    return { error: 'Chantier non trouvé.' };
  }

  // Chercher d'abord un worker déjà assigné à ce chantier
  const { data: existingWorker, error: workerFetchError } = await admin
    .from('workers')
    .select('id, name, role')
    .eq('site_id', siteId)
    .eq('email', email)
    .maybeSingle();

  if (workerFetchError && workerFetchError.code !== 'PGRST116') {
    // PGRST116 = not found, ce qui est OK
    return { error: workerFetchError.message };
  }

  let workerId = existingWorker?.id;

  // Si pas trouvé dans le chantier, chercher au niveau du compte (workers réutilisables)
  if (!workerId && site.created_by) {
    const { data: accountWorker } = await admin
      .from('workers')
      .select('id, name, role, created_by')
      .eq('created_by', site.created_by)
      .is('site_id', null)
      .eq('email', email)
      .maybeSingle();

    if (accountWorker) {
      // Worker trouvé au niveau du compte, créer une copie pour ce chantier
      const { data: newSiteWorker, error: workerInsertError } = await admin
        .from('workers')
        .insert({
          site_id: siteId,
          name: accountWorker.name || name || email,
          email,
          role: accountWorker.role || role || null,
        })
        .select('id')
        .single();

      if (workerInsertError || !newSiteWorker) {
        return { error: workerInsertError?.message ?? 'Impossible de créer le collaborateur pour ce chantier.' };
      }
      workerId = newSiteWorker.id;
    }
  }

  // Si toujours pas de worker, créer un nouveau worker pour ce chantier
  if (!workerId) {
    const { data: newWorker, error: workerInsertError } = await admin
      .from('workers')
      .insert({
        site_id: siteId,
        name: name || email,
        email,
        role: role || null,
      })
      .select('id')
      .single();

    if (workerInsertError || !newWorker) {
      return { error: workerInsertError?.message ?? 'Impossible de créer le collaborateur.' };
    }
    workerId = newWorker.id;
  }

  let photoUrl: string | null = null;

  if (file && file.size > 0) {
    const fileExt = file.name?.split('.').pop() ?? 'jpg';
    const filename = `site-${siteId}/${crypto.randomUUID()}.${fileExt}`;
    const { error: uploadError } = await admin.storage
      .from(REPORT_BUCKET)
      .upload(filename, file, {
        contentType: file.type || 'image/jpeg',
        upsert: false,
      });

    if (uploadError) {
      return { error: uploadError.message };
    }

    const {
      data: { publicUrl },
    } = admin.storage.from(REPORT_BUCKET).getPublicUrl(filename);
    photoUrl = publicUrl;
  }

  const { error: reportError } = await admin.from('reports').insert({
    task_id: taskId,
    worker_id: workerId,
    description: description || null,
    photo_url: photoUrl,
  });

  if (reportError) {
    return { error: reportError.message };
  }

  if (markDone) {
    await admin.from('tasks').update({ status: 'done' }).eq('id', taskId);
  }

  // Récupérer les infos du chantier et du manager pour l'email
  const { data: task } = await admin
    .from('tasks')
    .select('title')
    .eq('id', taskId)
    .single();

  const { data: siteInfo } = await admin
    .from('sites')
    .select('id, name, created_by')
    .eq('id', siteId)
    .single();

  if (siteInfo && task) {
    // Récupérer l'email du manager
    const { data: manager } = await admin.auth.admin.getUserById(siteInfo.created_by);
    
    if (manager?.user?.email) {
      const workerName = name || email;
      const appUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? '';
      const reportUrl = `${appUrl}/reports`;

      try {
        await sendReportNotificationEmail({
          managerEmail: manager.user.email,
          managerName: manager.user.user_metadata?.full_name || undefined,
          workerName,
          workerEmail: email || undefined,
          taskTitle: task.title,
          siteName: siteInfo.name,
          reportUrl,
        });
      } catch (error) {
        // Ne pas bloquer l'envoi du rapport si l'email échoue
        console.error('Erreur envoi email notification:', error);
      }
    }
  }

  revalidatePath(`/qr/${siteId}`);
  revalidatePath(`/site/${siteId}`);
  revalidatePath(`/report/${siteId}`);

  return { success: true };
}

