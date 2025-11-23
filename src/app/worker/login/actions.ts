'use server';

import { cookies } from 'next/headers';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export type WorkerLoginState = {
  error?: string;
  success?: boolean;
  workerId?: string;
  siteId?: string;
};

/**
 * Action pour connecter un worker avec son code d'accès
 */
export async function workerLoginAction(
  _prevState: WorkerLoginState,
  formData: FormData,
): Promise<WorkerLoginState> {
  const accessCode = String(formData.get('access_code') ?? '').trim().toUpperCase();
  const siteIdFromUrl = String(formData.get('siteId') ?? '').trim();

  if (!accessCode) {
    return { error: 'Code d\'accès requis.' };
  }

  // Vérifier le format du code (4 chiffres + 4 lettres)
  if (!/^[0-9]{4}[A-Z]{4}$/.test(accessCode)) {
    return { error: 'Format de code invalide. Format attendu: 4 chiffres + 4 lettres (ex: 1234ABCD).' };
  }

  const supabase = await createSupabaseServerClient();

  // Rechercher le worker uniquement par code d'accès
  const { data: worker, error: workerError } = await supabase
    .from('workers')
    .select('id, name, email, site_id, access_code')
    .eq('access_code', accessCode)
    .single();

  if (workerError || !worker) {
    return { error: 'Code d\'accès incorrect.' };
  }

  // Déterminer le siteId final
  let finalSiteId: string | null = null;

  // Si un siteId est fourni dans l'URL, vérifier que le worker peut accéder à ce chantier
  if (siteIdFromUrl) {
    // Vérifier si le worker est assigné à ce chantier directement
    if (worker.site_id === siteIdFromUrl) {
      finalSiteId = siteIdFromUrl;
    } else {
      // Vérifier si le worker est assigné à une tâche de ce chantier
      const { data: task } = await supabase
        .from('tasks')
        .select('site_id')
        .eq('site_id', siteIdFromUrl)
        .eq('assigned_worker_id', worker.id)
        .limit(1)
        .maybeSingle();

      if (task) {
        // Le worker est assigné à une tâche de ce chantier
        finalSiteId = siteIdFromUrl;
      } else {
        return { error: 'Vous n\'êtes pas assigné à ce chantier.' };
      }
    }
  } else {
    // Si pas de siteId dans l'URL, chercher un chantier où le worker a une tâche assignée
    // Priorité: site_id du worker, puis tâche assignée
    if (worker.site_id) {
      finalSiteId = worker.site_id;
    } else {
      // Worker au niveau du compte, chercher une tâche assignée
      const { data: task } = await supabase
        .from('tasks')
        .select('site_id')
        .eq('assigned_worker_id', worker.id)
        .limit(1)
        .maybeSingle();

      if (task) {
        finalSiteId = task.site_id;
      } else {
        return { error: 'Aucune tâche assignée. Contactez votre responsable.' };
      }
    }
  }

  if (!finalSiteId) {
    return { error: 'Aucun chantier trouvé. Contactez votre responsable.' };
  }

  // Créer un cookie de session pour le worker
  const cookieStore = await cookies();
  cookieStore.set('worker_session', JSON.stringify({
    workerId: worker.id,
    siteId: finalSiteId,
    email: worker.email,
    name: worker.name,
  }), {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 7, // 7 jours
    path: '/',
  });

  return {
    success: true,
    workerId: worker.id,
    siteId: finalSiteId,
  };
}

