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

  // Vérifier que le worker a un site_id (assigné à un chantier)
  if (!worker.site_id) {
    return { error: 'Aucun chantier assigné. Contactez votre responsable.' };
  }

  // Créer un cookie de session pour le worker
  const cookieStore = await cookies();
  cookieStore.set('worker_session', JSON.stringify({
    workerId: worker.id,
    siteId: worker.site_id,
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
    siteId: worker.site_id,
  };
}

