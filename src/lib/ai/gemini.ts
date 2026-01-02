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
  // Liste des modèles à essayer dans l'ordre de préférence
  const modelsToTry = ['gemini-2.0-flash-exp', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-pro'];
  
  const client = getGeminiClient();
  let lastError: Error | null = null;
  
  for (const modelName of modelsToTry) {
    try {
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

      // Si on arrive ici, le modèle fonctionne
      return text;
    } catch (error) {
      // Si c'est une erreur 404 (modèle non trouvé), essayer le suivant
      if (error instanceof Error && (error.message.includes('not found') || error.message.includes('404'))) {
        console.warn(`[Gemini] Modèle ${modelName} non disponible, tentative avec le suivant...`);
        lastError = error;
        continue; // Essayer le modèle suivant
      }
      // Si c'est une autre erreur, la propager
      throw error;
    }
  }
  
  // Si aucun modèle n'a fonctionné
  throw new Error(`Aucun modèle Gemini disponible. Dernière erreur: ${lastError?.message || 'Inconnue'}`);
}

/**
 * Vérifie si Gemini est configuré
 */
export async function isGeminiConfigured(): Promise<boolean> {
  const apiKey = process.env.GOOGLE_GEMINI_API_KEY;
  return !!apiKey && apiKey.trim() !== '';
}
