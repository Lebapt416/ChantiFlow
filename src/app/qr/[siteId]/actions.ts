'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';

type ReportState = {
  error?: string;
  success?: boolean;
};

const REPORT_BUCKET = 'reports';

export async function submitReportAction(
  _prevState: ReportState,
  formData: FormData,
): Promise<ReportState> {
  const siteId = String(formData.get('siteId') ?? '');
  const taskId = String(formData.get('taskId') ?? '');
  const email = String(formData.get('email') ?? '').trim().toLowerCase();
  const name = String(formData.get('name') ?? '').trim();
  const role = String(formData.get('role') ?? '').trim();
  const description = String(formData.get('description') ?? '').trim();
  const markDone = formData.get('mark_done') === 'on';
  const file = formData.get('photo') as File | null;

  if (!siteId || !taskId || !email) {
    return { error: 'Site, tâche et email sont requis.' };
  }

  const admin = createSupabaseAdminClient();

  // upsert worker
  const { data: existingWorker, error: workerFetchError } = await admin
    .from('workers')
    .select('id')
    .eq('site_id', siteId)
    .eq('email', email)
    .maybeSingle();

  if (workerFetchError) {
    return { error: workerFetchError.message };
  }

  let workerId = existingWorker?.id;

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

  revalidatePath(`/qr/${siteId}`);
  revalidatePath(`/site/${siteId}`);
  revalidatePath(`/report/${siteId}`);

  return { success: true };
}

