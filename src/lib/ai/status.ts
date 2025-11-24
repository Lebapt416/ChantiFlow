'use server';

type AIStatus =
  | { status: 'missing_config'; message: string }
  | { status: 'connected'; message: string; url: string }
  | { status: 'error'; message: string };

function normalizeUrl(rawUrl: string) {
  const trimmed = rawUrl.trim().replace(/\/$/, '');
  if (!trimmed.startsWith('http://') && !trimmed.startsWith('https://')) {
    return `https://${trimmed}`;
  }
  return trimmed;
}

export async function checkAIStatus(): Promise<AIStatus> {
  const baseUrl = process.env.NEXT_PUBLIC_PREDICTION_API_URL?.trim();

  if (!baseUrl) {
    return { status: 'missing_config', message: 'URL non configurée' };
  }

  const normalizedUrl = normalizeUrl(baseUrl);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${normalizedUrl}/health`, {
      method: 'GET',
      signal: controller.signal,
      cache: 'no-store',
    });

    if (response.ok) {
      return { status: 'connected', message: 'IA Connectée', url: normalizedUrl };
    }

    return { status: 'error', message: 'Erreur de connexion' };
  } catch {
    return { status: 'error', message: 'Erreur de connexion' };
  } finally {
    clearTimeout(timeoutId);
  }
}

