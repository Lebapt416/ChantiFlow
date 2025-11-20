'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';

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
  const name = String(formData.get('name') ?? '').trim();
  const email = String(formData.get('email') ?? '').trim();
  const role = String(formData.get('role') ?? '').trim();

  if (!siteId || !name) {
    return { error: 'Site et nom requis.' };
  }

  const supabase = await createSupabaseServerClient();
  const { error } = await supabase.from('workers').insert({
    site_id: siteId,
    name,
    email: email || null,
    role: role || null,
  });

  if (error) {
    return { error: error.message };
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

