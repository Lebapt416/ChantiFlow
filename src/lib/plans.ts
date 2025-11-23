import { createSupabaseServerClient } from '@/lib/supabase/server';

export type Plan = 'basic' | 'plus' | 'pro';

export type AddOn = {
  type: 'extra_workers' | 'extra_sites';
  quantity: number; // Nombre d'add-ons achetés
};

export interface UserAddOns {
  extra_workers?: number; // Nombre d'add-ons "+5 employés" achetés
  extra_sites?: number; // Nombre d'add-ons "+2 chantiers" achetés
}

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

export function getUserAddOns(user: {
  user_metadata?: Record<string, unknown>;
}): UserAddOns {
  const addOns = user.user_metadata?.addOns as UserAddOns | undefined;
  return {
    extra_workers: addOns?.extra_workers ?? 0,
    extra_sites: addOns?.extra_sites ?? 0,
  };
}

export function getPlanLimits(plan: Plan, addOns?: UserAddOns): { maxSites: number; maxWorkers: number } {
  const baseLimits = {
    basic: { maxSites: 1, maxWorkers: 3 },
    plus: { maxSites: 5, maxWorkers: 7 },
    pro: { maxSites: Infinity, maxWorkers: Infinity },
  };

  const limits = baseLimits[plan] ?? baseLimits.basic;

  // Ajouter les add-ons
  if (addOns) {
    return {
      maxSites: limits.maxSites === Infinity 
        ? Infinity 
        : limits.maxSites + (addOns.extra_sites ?? 0) * 2, // +2 chantiers par add-on
      maxWorkers: limits.maxWorkers === Infinity 
        ? Infinity 
        : limits.maxWorkers + (addOns.extra_workers ?? 0) * 5, // +5 employés par add-on
    };
  }

  return limits;
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
  const addOns = getUserAddOns(user);
  const limits = getPlanLimits(plan, addOns);

  if (limits.maxSites === Infinity) {
    return { allowed: true };
  }

  // Compter uniquement les chantiers actifs (non terminés)
  const { data: sites } = await supabase
    .from('sites')
    .select('id, completed_at')
    .eq('created_by', userId);

  // Filtrer les chantiers terminés
  const activeSites = sites?.filter((site) => !site.completed_at) ?? [];
  const siteCount = activeSites.length;

  if (siteCount >= limits.maxSites) {
    return {
      allowed: false,
      reason: `Limite atteinte (${limits.maxSites} chantier${limits.maxSites > 1 ? 's' : ''} max pour le plan ${plan}${addOns.extra_sites ? ` + ${addOns.extra_sites} add-on(s)` : ''}). Passez au plan supérieur ou ajoutez un add-on.`,
    };
  }

  return { allowed: true };
}

export async function canAddWorker(userId: string): Promise<{ allowed: boolean; reason?: string }> {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { allowed: false, reason: 'Non authentifié' };
  }

  const plan = await getUserPlan(user);
  const addOns = getUserAddOns(user);
  const limits = getPlanLimits(plan, addOns);

  if (limits.maxWorkers === Infinity) {
    return { allowed: true };
  }

  // Compter tous les workers au niveau du compte (sans site_id) et ceux assignés à des chantiers actifs
  const { data: accountWorkers } = await supabase
    .from('workers')
    .select('id')
    .eq('created_by', userId)
    .is('site_id', null);

  // Récupérer les chantiers actifs
  const { data: activeSites } = await supabase
    .from('sites')
    .select('id')
    .eq('created_by', userId)
    .is('completed_at', null);

  const activeSiteIds = activeSites?.map((s) => s.id) ?? [];

  // Compter les workers assignés aux chantiers actifs
  const { data: siteWorkers } = activeSiteIds.length > 0
    ? await supabase
        .from('workers')
        .select('id')
        .in('site_id', activeSiteIds)
    : { data: [] };

  // Total = workers au niveau compte + workers sur chantiers actifs
  const totalWorkers = (accountWorkers?.length ?? 0) + (siteWorkers?.length ?? 0);

  if (totalWorkers >= limits.maxWorkers) {
    return {
      allowed: false,
      reason: `Limite atteinte (${limits.maxWorkers} employé${limits.maxWorkers > 1 ? 's' : ''} max pour le plan ${plan}${addOns.extra_workers ? ` + ${addOns.extra_workers} add-on(s)` : ''}). Passez au plan supérieur ou ajoutez un add-on.`,
    };
  }

  return { allowed: true };
}

