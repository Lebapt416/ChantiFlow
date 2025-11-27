'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { readWorkerSession, updateWorkerSession } from '@/lib/worker-session';

type JoinSiteResult = {
  success?: boolean;
  error?: string;
  siteId?: string;
  siteName?: string | null;
};

function extractSiteId(rawValue: string) {
  if (!rawValue) return null;
  const trimmed = rawValue.trim();

  const uuidRegex = /^[0-9a-fA-F-]{32,36}$/;
  if (uuidRegex.test(trimmed)) {
    return trimmed.toLowerCase();
  }

  try {
    const url = new URL(trimmed);
    const segments = url.pathname.split('/').filter(Boolean);
    if (segments.length) {
      const candidate = segments[segments.length - 1];
      if (uuidRegex.test(candidate)) {
        return candidate.toLowerCase();
      }
    }
    if (url.protocol === 'chantiflow:' && segments[0] === 'site') {
      const candidate = segments[1];
      if (candidate && uuidRegex.test(candidate)) {
        return candidate.toLowerCase();
      }
    }
  } catch {
    // Not a valid URL, continue
  }

  if (trimmed.startsWith('chantiflow://')) {
    const afterScheme = trimmed.replace('chantiflow://', '');
    const parts = afterScheme.split('/').filter(Boolean);
    const candidate = parts[parts.length - 1];
    if (candidate && uuidRegex.test(candidate)) {
      return candidate.toLowerCase();
    }
  }

  return null;
}

export async function joinSiteAction(scanValue: string): Promise<JoinSiteResult> {
  const session = await readWorkerSession();
  if (!session?.workerId) {
    return { error: 'Session expirée. Veuillez vous reconnecter.' };
  }

  const siteId = extractSiteId(scanValue);
  if (!siteId) {
    return { error: 'QR code invalide. Scan d\'un chantier ChantiFlow requis.' };
  }

  const supabase = await createSupabaseServerClient();
  const { data: worker, error: workerError } = await supabase
    .from('workers')
    .select('id, created_by')
    .eq('id', session.workerId)
    .single();

  if (workerError || !worker) {
    return { error: 'Employé introuvable.' };
  }

  const { data: site, error: siteError } = await supabase
    .from('sites')
    .select('id, name, created_by')
    .eq('id', siteId)
    .single();

  if (siteError || !site) {
    return { error: 'Chantier introuvable.' };
  }

  if (worker.created_by && site.created_by && worker.created_by !== site.created_by) {
    return { error: 'Ce QR code appartient à une autre entreprise.' };
  }

  const { error: updateError } = await supabase
    .from('workers')
    .update({ site_id: site.id })
    .eq('id', worker.id);

  if (updateError) {
    return { error: 'Impossible d\'assigner ce chantier. Réessayez.' };
  }

  await updateWorkerSession({ siteId: site.id });
  revalidatePath('/worker/dashboard');
  revalidatePath(`/worker/${site.id}`);

  return {
    success: true,
    siteId: site.id,
    siteName: site.name,
  };
}

