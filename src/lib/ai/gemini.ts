'use server';

import { GoogleGenerativeAI } from '@google/generative-ai';

/**
 * Initialise le client Google Gemini
 */
function getGeminiClient() {
  const apiKey = process.env.GOOGLE_GEMINI_API_KEY;

  if (!apiKey || apiKey.trim() === '') {
    throw new Error('GOOGLE_GEMINI_API_KEY non configurée');
  }

  return new GoogleGenerativeAI(apiKey);
}

/**
 * Génère une réponse avec Google Gemini
 * @param prompt - Le prompt utilisateur
 * @param systemInstruction - Instruction système (optionnel)
 * @param options - Options supplémentaires (temperature, maxTokens, etc.)
 */
export async function generateWithGemini(
  prompt: string,
  systemInstruction?: string,
  options?: {
    temperature?: number;
    maxOutputTokens?: number;
    responseFormat?: 'json' | 'text';
  },
): Promise<string> {
  try {
    const client = getGeminiClient();
    // Utiliser gemini-pro qui est le modèle le plus compatible avec l'API v1beta
    const modelName = 'gemini-pro';
    const model = client.getGenerativeModel({
      model: modelName,
      systemInstruction: systemInstruction || 'Tu es un assistant expert en gestion de chantiers de construction en France.',
    });

    const generationConfig: {
      temperature?: number;
      maxOutputTokens?: number;
      responseMimeType?: string;
    } = {};

    if (options?.temperature !== undefined) {
      generationConfig.temperature = options.temperature;
    }

    if (options?.maxOutputTokens !== undefined) {
      generationConfig.maxOutputTokens = options.maxOutputTokens;
    }

    // Si on veut du JSON, on le spécifie
    if (options?.responseFormat === 'json') {
      generationConfig.responseMimeType = 'application/json';
    }

    const result = await model.generateContent({
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      generationConfig,
    });

    const response = result.response;
    const text = response.text();

    if (!text) {
      throw new Error('Réponse Gemini vide');
    }

    return text;
  } catch (error) {
    console.error('[Gemini] Erreur:', error);
    
    if (error instanceof Error) {
      // Gestion spécifique des erreurs Gemini
      if (error.message.includes('API_KEY_INVALID')) {
        throw new Error('Clé API Google Gemini invalide. Vérifiez votre clé sur console.cloud.google.com');
      }
      if (error.message.includes('QUOTA_EXCEEDED') || error.message.includes('429')) {
        throw new Error('Quota Google Gemini dépassé. Attendez quelques minutes ou vérifiez votre quota sur console.cloud.google.com');
      }
      if (error.message.includes('SAFETY')) {
        throw new Error('Contenu bloqué par les filtres de sécurité Gemini');
      }
      // Si le modèle n'est pas trouvé, fournir un message d'erreur plus clair
      if (error.message.includes('not found') || error.message.includes('404')) {
        throw new Error(`Modèle Gemini 'gemini-pro' non disponible avec l'API v1beta. Vérifiez que votre clé API a accès aux modèles Gemini et que le modèle est disponible dans votre région. Erreur: ${error.message}`);
      }
    }
    
    throw error;
  }
}

/**
 * Vérifie si Gemini est configuré
 */
export async function isGeminiConfigured(): Promise<boolean> {
  const apiKey = process.env.GOOGLE_GEMINI_API_KEY;
  return !!apiKey && apiKey.trim() !== '';
}

