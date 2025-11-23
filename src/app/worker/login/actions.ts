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
  const email = String(formData.get('email') ?? '').trim().toLowerCase();
  const accessCode = String(formData.get('access_code') ?? '').trim().toUpperCase();
  const siteIdFromUrl = String(formData.get('siteId') ?? '').trim();

  if (!email || !accessCode) {
    return { error: 'Email et code d\'accès requis.' };
  }

  const supabase = await createSupabaseServerClient();

  // Rechercher le worker par email et code d'accès
  const { data: worker, error: workerError } = await supabase
    .from('workers')
    .select('id, name, email, site_id, access_code')
    .eq('email', email)
    .eq('access_code', accessCode)
    .single();

  if (workerError || !worker) {
    return { error: 'Code d\'accès ou email incorrect.' };
  }

  // Si un siteId est fourni dans l'URL, vérifier que le worker est assigné à ce chantier
  let finalSiteId = worker.site_id;
  if (siteIdFromUrl) {
    // Vérifier que le worker est bien assigné à ce chantier
    if (worker.site_id !== siteIdFromUrl) {
      // Peut-être que le worker est au niveau du compte, vérifier s'il peut accéder à ce chantier
      const { data: siteWorker } = await supabase
        .from('workers')
        .select('id, site_id')
        .eq('id', worker.id)
        .eq('site_id', siteIdFromUrl)
        .single();
      
      if (!siteWorker) {
        return { error: 'Vous n\'êtes pas assigné à ce chantier.' };
      }
      finalSiteId = siteIdFromUrl;
    }
  } else {
    // Si pas de siteId dans l'URL, vérifier que le worker a un site_id
    if (!worker.site_id) {
      return { error: 'Aucun chantier assigné. Contactez votre responsable.' };
    }
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

