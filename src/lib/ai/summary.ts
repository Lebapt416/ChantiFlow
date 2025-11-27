'use server';

function normalizeUrl(rawUrl: string | undefined): string | null {
  if (!rawUrl) return null;
  const trimmed = rawUrl.trim().replace(/\/$/, '');
  if (!trimmed.startsWith('http://') && !trimmed.startsWith('https://')) {
    return `https://${trimmed}`;
  }
  return trimmed;
}

const API_URL = normalizeUrl(process.env.NEXT_PUBLIC_PREDICTION_API_URL || process.env.ML_API_URL);

export type AISummaryResponse = {
  summary: string;
  status: string;
  sites_mentioned?: string[];
};

export async function generateGlobalSummary(sites: any[]): Promise<AISummaryResponse | null> {
  if (!API_URL) {
    console.warn('⚠️ URL API non configurée pour les résumés');
    return null;
  }
  
  try {
    // Transformer les données pour l'API Python
    const payload = {
      sites: sites.map(s => ({
        name: s.name,
        tasks_total: s.tasks_total || s.tasks?.length || 0,
        tasks_done: s.tasks_done || s.tasks?.filter((t: any) => t.status === 'done').length || 0,
        complexity: s.complexity || 5.0, // Utiliser la complexité fournie ou valeur par défaut
        days_elapsed: s.days_elapsed || Math.ceil((new Date().getTime() - new Date(s.created_at).getTime()) / (1000 * 3600 * 24))
      }))
    };

    const res = await fetch(`${API_URL}/summary/global`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      cache: 'no-store',
    });
    
    if (!res.ok) {
      const errorText = await res.text();
      console.error(`❌ Erreur API résumé global: ${res.status} - ${errorText}`);
      return null;
    }
    
    return await res.json() as AISummaryResponse;
    
  } catch (e) {
    console.error("❌ Erreur résumé global:", e);
    return null;
  }
}

export async function generateSiteSummary(site: any, tasks: any[], hasWeatherAccess: boolean = false): Promise<AISummaryResponse | null> {
  if (!API_URL) {
    console.warn('⚠️ URL API non configurée pour les résumés. Variables:', {
      NEXT_PUBLIC_PREDICTION_API_URL: process.env.NEXT_PUBLIC_PREDICTION_API_URL,
      ML_API_URL: process.env.ML_API_URL
    });
    return null;
  }
  
  console.log('🌐 API URL configurée:', API_URL);

  try {
    const totalTasks = tasks.length;
    const pendingTasks = tasks.filter(t => t.status === 'pending').length;
    
    // Calcul de la complexité basé sur la diversité des rôles et la durée moyenne
    const roleDiversity = new Set(tasks.map((t: any) => t.required_role).filter(Boolean)).size;
    const avgDuration = tasks.reduce((sum: number, t: any) => sum + (t.duration_hours || 8), 0) / Math.max(1, totalTasks);
    const complexity = Math.min(10, Math.max(1, avgDuration / 4 + roleDiversity / 2));
    
    // Estimation deadline théorique
    const plannedDays = site.deadline 
      ? Math.ceil((new Date(site.deadline).getTime() - new Date(site.created_at).getTime()) / (1000 * 3600 * 24))
      : 30;

    const payload = {
      site_name: site.name,
      tasks_total: totalTasks,
      tasks_pending: pendingTasks,
      complexity: Number(complexity.toFixed(2)),
      days_elapsed: Math.ceil((new Date().getTime() - new Date(site.created_at).getTime()) / (1000 * 3600 * 24)),
      planned_duration: plannedDays,
      // Inclure le code postal seulement si l'utilisateur a accès à la météo
      location: (hasWeatherAccess && site.postal_code) ? site.postal_code.trim() : null
    };

    console.log('📤 Envoi résumé IA:', {
      site_name: payload.site_name,
      tasks_total: payload.tasks_total,
      hasWeatherAccess,
      postal_code: site.postal_code || 'non défini',
      location: payload.location || 'non fourni',
      api_url: API_URL
    });

    const res = await fetch(`${API_URL}/summary/site`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      cache: 'no-store',
    });

    if (!res.ok) {
      const errorText = await res.text();
      console.error(`❌ Erreur API résumé site: ${res.status} - ${errorText}`);
      return null;
    }
    
    const result = await res.json();
    console.log('✅ Résumé IA reçu:', { summary: result.summary?.substring(0, 100) + '...', status: result.status });
    return result;

  } catch (e) {
    console.error("❌ Erreur résumé chantier:", e);
    return null;
  }
}
