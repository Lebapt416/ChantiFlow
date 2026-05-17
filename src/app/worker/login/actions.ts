'use server';

import { randomUUID } from 'crypto';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';
import { writeWorkerSession } from '@/lib/worker-session';

export type WorkerLoginState = {
  error?: string;
  success?: boolean;
  workerId?: string;
  siteId?: string | null;
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

  // service_role pour bypass RLS — le worker n'est pas encore authentifié
  // mais le code d'accès lui-même fait office de credential
  const admin = createSupabaseAdminClient();
  const { data: worker, error: workerError } = await admin
    .from('workers')
    .select('id, name, email, site_id, access_code, access_token')
    .eq('access_code', accessCode)
    .single();

  if (workerError || !worker) {
    return { error: 'Code d\'accès incorrect.' };
  }

  // Client anon pour les tables avec RLS publique (tasks)
  const supabase = await createSupabaseServerClient();

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

  // S'assurer qu'un access_token est présent
  let accessToken = worker.access_token as string | null;
  const updates: Record<string, unknown> = {
    last_login: new Date().toISOString(),
  };
  if (!accessToken) {
    accessToken = randomUUID();
    updates.access_token = accessToken;
  }
  await admin.from('workers').update(updates).eq('id', worker.id);

  await writeWorkerSession({
    workerId: worker.id,
    token: accessToken,
    siteId: finalSiteId,
    email: worker.email,
    name: worker.name,
  });

  return {
    success: true,
    workerId: worker.id,
    siteId: finalSiteId,
  };
}

export async function loginWorkerWithToken(token: string): Promise<WorkerLoginState> {
  const sanitizedToken = token?.trim();
  if (!sanitizedToken) {
    return { error: 'Token manquant.' };
  }

  // service_role pour bypass RLS (même raison que workerLoginAction)
  const admin = createSupabaseAdminClient();
  const { data: worker, error } = await admin
    .from('workers')
    .select('id, name, email, site_id, access_token')
    .eq('access_token', sanitizedToken)
    .single();

  if (error || !worker) {
    return { error: 'Lien invalide ou expiré.' };
  }

  await admin
    .from('workers')
    .update({ last_login: new Date().toISOString() })
    .eq('id', worker.id);

  await writeWorkerSession({
    workerId: worker.id,
    token: worker.access_token,
    siteId: worker.site_id,
    email: worker.email,
    name: worker.name,
  });

  return {
    success: true,
    workerId: worker.id,
    siteId: worker.site_id,
  };
}

