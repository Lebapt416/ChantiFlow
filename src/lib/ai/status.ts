'use server';

import { isGeminiConfigured } from './gemini';

type AIStatus =
  | { status: 'missing_config'; message: string }
  | { status: 'connected'; message: string; url?: string }
  | { status: 'error'; message: string };

export async function checkAIStatus(): Promise<AIStatus> {
  // Vérifier d'abord Google Gemini (priorité)
  const geminiConfigured = await isGeminiConfigured();
  
  if (geminiConfigured) {
    return { 
      status: 'connected', 
      message: 'Google Gemini connecté',
      url: 'Google Gemini API'
    };
  }

  // Fallback : vérifier l'ancienne API de prédiction (si configurée)
  const baseUrl = process.env.NEXT_PUBLIC_PREDICTION_API_URL?.trim();

  if (!baseUrl) {
    return { 
      status: 'missing_config', 
      message: 'Google Gemini non configuré. Ajoutez GOOGLE_GEMINI_API_KEY dans Vercel.' 
    };
  }

  // Tester l'API de prédiction si elle est configurée
  function normalizeUrl(rawUrl: string) {
    const trimmed = rawUrl.trim().replace(/\/$/, '');
    if (!trimmed.startsWith('http://') && !trimmed.startsWith('https://')) {
      return `https://${trimmed}`;
    }
    return trimmed;
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

    return { status: 'error', message: 'Erreur de connexion à l\'API de prédiction' };
  } catch {
    return { status: 'error', message: 'Erreur de connexion à l\'API de prédiction' };
  } finally {
    clearTimeout(timeoutId);
  }
}

