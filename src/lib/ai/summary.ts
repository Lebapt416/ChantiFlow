const API_BASE_URL = process.env.NEXT_PUBLIC_PREDICTION_API_URL || 'http://localhost:8000';

export type SummaryStatus = 'good' | 'warning' | 'critical';

export type SummaryResponse = {
  summary: string;
  status: SummaryStatus;
};

export type GlobalSummarySite = {
  name: string;
  tasks_total: number;
  tasks_done: number;
  complexity: number;
  days_elapsed: number;
};

export type SiteSummaryPayload = {
  site_name: string;
  tasks_total: number;
  tasks_pending: number;
  complexity: number;
  days_elapsed: number;
  planned_duration: number;
};

async function postSummary<TInput>(
  endpoint: string,
  body: TInput,
  fallback: SummaryResponse,
): Promise<SummaryResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(body),
      cache: 'no-store',
    });

    if (!response.ok) {
      const detail = await response.text();
      console.warn('Résumé IA indisponible:', detail);
      return fallback;
    }

    return (await response.json()) as SummaryResponse;
  } catch (error) {
    console.error('Erreur appel résumé IA:', error);
    return fallback;
  }
}

export async function generateGlobalSummary(
  sites: GlobalSummarySite[],
): Promise<SummaryResponse> {
  return postSummary('/summary/global', { sites }, {
    summary: sites.length
      ? 'Analyse IA indisponible. Surveillance manuelle recommandée.'
      : 'Aucun chantier actif pour le moment.',
    status: sites.length ? 'warning' : 'good',
  });
}

export async function generateSiteSummary(
  payload: SiteSummaryPayload,
): Promise<SummaryResponse> {
  return postSummary('/summary/site', payload, {
    summary: 'Analyse IA indisponible. Vérifiez manuellement l’avancement.',
    status: 'warning',
  });
}

