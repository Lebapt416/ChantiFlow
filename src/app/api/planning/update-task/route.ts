import { NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server';

/**
 * PATCH /api/planning/update-task
 * Persiste le déplacement d'une tâche dans la vue Gantt.
 * Appelé après drag & drop ou redimensionnement.
 *
 * Body: { siteId: string, taskId: string, startDate: string, endDate: string }
 */
export async function PATCH(request: Request) {
  const supabase = await createSupabaseServerClient();

  // Auth
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return NextResponse.json({ error: 'Non authentifié.' }, { status: 401 });
  }

  let body: { siteId?: string; taskId?: string; startDate?: string; endDate?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Corps de requête invalide.' }, { status: 400 });
  }

  const { siteId, taskId, startDate, endDate } = body;

  if (!siteId || !taskId || !startDate || !endDate) {
    return NextResponse.json({ error: 'Paramètres manquants.' }, { status: 400 });
  }

  // Vérifier que l'utilisateur est propriétaire du chantier
  const { data: site } = await supabase
    .from('sites')
    .select('id, created_by')
    .eq('id', siteId)
    .single();

  if (!site || site.created_by !== user.id) {
    return NextResponse.json({ error: 'Accès refusé.' }, { status: 403 });
  }

  // Vérifier que la tâche appartient à ce chantier
  const { data: task } = await supabase
    .from('tasks')
    .select('id, site_id')
    .eq('id', taskId)
    .eq('site_id', siteId)
    .single();

  if (!task) {
    return NextResponse.json({ error: 'Tâche introuvable.' }, { status: 404 });
  }

  // Mettre à jour les dates
  const { error } = await supabase
    .from('tasks')
    .update({
      planned_start: startDate,
      planned_end: endDate,
    })
    .eq('id', taskId)
    .eq('site_id', siteId);

  if (error) {
    console.error('[planning/update-task] Supabase error:', error);
    return NextResponse.json({ error: 'Erreur lors de la mise à jour.' }, { status: 500 });
  }

  return NextResponse.json({ ok: true });
}
