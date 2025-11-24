'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';

type UpdateTaskPlanningPayload = {
  siteId: string;
  taskId: string;
  plannedStart: string;
  plannedEnd: string;
  durationHours?: number;
  workerId?: string | null;
  plannedOrder?: number;
};

export async function updateTaskPlanningAction({
  siteId,
  taskId,
  plannedStart,
  plannedEnd,
  durationHours,
  workerId,
  plannedOrder,
}: UpdateTaskPlanningPayload) {
  const supabase = await createSupabaseServerClient();

  const updatePayload: Record<string, any> = {
    planned_start: plannedStart,
    planned_end: plannedEnd,
    planned_worker_id: workerId ?? null,
  };

  if (typeof plannedOrder === 'number') {
    updatePayload.planned_order = plannedOrder;
  }

  if (typeof durationHours === 'number' && Number.isFinite(durationHours)) {
    updatePayload.duration_hours = durationHours;
    const startDate = new Date(plannedStart);
    const endDate = new Date(startDate.getTime() + durationHours * 60 * 60 * 1000);
    updatePayload.planned_end = endDate.toISOString();
  }

  await supabase.from('tasks').update(updatePayload).eq('id', taskId);

  revalidatePath(`/site/${siteId}/planning`);
  revalidatePath(`/site/${siteId}/dashboard`);
  revalidatePath(`/site/${siteId}`);
}

