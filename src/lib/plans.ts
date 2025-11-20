'use server';

import { createSupabaseServerClient } from '@/lib/supabase/server';

export type Plan = 'basic' | 'plus' | 'pro';

export async function getUserPlan(user: {
  id: string;
  user_metadata?: Record<string, unknown>;
}): Promise<Plan> {
  // Lire le plan depuis les métadonnées utilisateur
  const planFromMetadata = user.user_metadata?.plan as Plan | undefined;

  if (planFromMetadata && ['basic', 'plus', 'pro'].includes(planFromMetadata)) {
    return planFromMetadata;
  }

  // Fallback : déterminer selon le nombre de chantiers
  const supabase = await createSupabaseServerClient();
  const { data: sites } = await supabase
    .from('sites')
    .select('id')
    .eq('created_by', user.id);

  const siteCount = sites?.length ?? 0;

  if (siteCount === 0) {
    return 'basic';
  } else if (siteCount <= 1) {
    return 'basic';
  } else if (siteCount <= 5) {
    return 'plus';
  } else {
    return 'pro';
  }
}

export function getPlanLimits(plan: Plan): { maxSites: number } {
  switch (plan) {
    case 'basic':
      return { maxSites: 1 };
    case 'plus':
      return { maxSites: 5 };
    case 'pro':
      return { maxSites: Infinity };
    default:
      return { maxSites: 1 };
  }
}

export async function canCreateSite(userId: string): Promise<{ allowed: boolean; reason?: string }> {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { allowed: false, reason: 'Non authentifié' };
  }

  const plan = await getUserPlan(user);
  const limits = getPlanLimits(plan);

  if (limits.maxSites === Infinity) {
    return { allowed: true };
  }

  const { data: sites } = await supabase
    .from('sites')
    .select('id')
    .eq('created_by', userId);

  const siteCount = sites?.length ?? 0;

  if (siteCount >= limits.maxSites) {
    return {
      allowed: false,
      reason: `Limite atteinte (${limits.maxSites} chantier${limits.maxSites > 1 ? 's' : ''} max pour le plan ${plan}). Passez au plan supérieur.`,
    };
  }

  return { allowed: true };
}

