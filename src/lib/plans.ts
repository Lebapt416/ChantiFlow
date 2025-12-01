import { createSupabaseServerClient } from '@/lib/supabase/server';

export type Plan = 'basic' | 'plus' | 'pro';

export const PLANS = {
  basic: {
    name: 'Basic',
    features: [
      '1 chantier actif',
      'Planification IA basique',
      'QR codes pour les employés',
      'Upload de photos et rapports',
      'Support par email',
    ],
  },
  plus: {
    name: 'Plus',
    features: [
      'Jusqu\'à 5 chantiers actifs',
      'Planification IA avancée',
      'Classement intelligent des tâches',
      'Analytics détaillés',
      'Support prioritaire',
      'Export de rapports',
      'Météo de chantier en temps réel',
    ],
  },
  pro: {
    name: 'Pro',
    features: [
      'Tout de Plus',
      'Multi-utilisateurs',
      'API personnalisée',
      'Intégrations avancées',
      'Support dédié 24/7',
      'Formation personnalisée',
      'Gestion des permissions',
      'Exports rapports PDF professionnels',
      'Assistant IA de chantier',
    ],
  },
} as const;

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
  const planType = user.user_metadata?.plan_type as 'monthly' | 'annual' | undefined;
  const planExpiresAt = user.user_metadata?.plan_expires_at as string | undefined;

  if (planFromMetadata && ['basic', 'plus', 'pro'].includes(planFromMetadata)) {
    // Vérifier si c'est un plan annuel et s'il est expiré
    if (planType === 'annual' && planExpiresAt) {
      const expirationDate = new Date(planExpiresAt);
      const now = new Date();
      
      if (now > expirationDate) {
        console.log(`Plan ${planFromMetadata} expiré pour l'utilisateur ${user.id}. Expiration: ${expirationDate.toISOString()}`);
        // Le plan est expiré, retourner 'basic'
        return 'basic';
      }
    }
    
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

/**
 * Vérifie si l'utilisateur a accès à la fonctionnalité météo (plan PLUS ou supérieur)
 */
export function canAccessWeather(planId: Plan): boolean {
  return planId === 'plus' || planId === 'pro';
}

/**
 * Vérifie si l'utilisateur a accès aux fonctionnalités PRO
 */
export function canAccessProFeatures(planId: Plan): boolean {
  return planId === 'pro';
}

