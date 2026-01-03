/**
 * Système de tests pour vérifier le bon fonctionnement de l'application
 */

export type TestResult = {
  name: string;
  status: 'success' | 'error' | 'warning';
  message: string;
  details?: string;
  duration?: number;
};

export type SystemTestReport = {
  overallScore: number; // Note sur 20
  totalTests: number;
  passedTests: number;
  failedTests: number;
  warningTests: number;
  tests: TestResult[];
  timestamp: string;
};

/**
 * Exécute tous les tests système
 */
export async function runSystemTests(): Promise<SystemTestReport> {
  const tests: TestResult[] = [];
  const startTime = Date.now();

  // Test 1: Variables d'environnement Supabase
  await testSupabaseEnv(tests);

  // Test 2: Connexion Supabase
  await testSupabaseConnection(tests);

  // Test 3: Accès aux tables principales
  await testDatabaseTables(tests);

  // Test 4: Configuration Gemini
  await testGeminiConfig(tests);

  // Test 5: Test Gemini API (si configuré)
  await testGeminiAPI(tests);

  // Test 6: Configuration de l'application
  await testAppConfig(tests);

  // Test 7: API ML (si configurée)
  await testMLAPI(tests);

  // Calcul de la note
  const passedTests = tests.filter(t => t.status === 'success').length;
  const failedTests = tests.filter(t => t.status === 'error').length;
  const warningTests = tests.filter(t => t.status === 'warning').length;
  const totalTests = tests.length;

  // Note sur 20 : 20 points si tout passe, -1 point par erreur, -0.5 point par warning
  const score = Math.max(0, 20 - failedTests - (warningTests * 0.5));

  return {
    overallScore: Math.round(score * 10) / 10,
    totalTests,
    passedTests,
    failedTests,
    warningTests,
    tests,
    timestamp: new Date().toISOString(),
  };
}

async function testSupabaseEnv(tests: TestResult[]) {
  const start = Date.now();
  try {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
    const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

    const missing: string[] = [];
    if (!url) missing.push('NEXT_PUBLIC_SUPABASE_URL');
    if (!anonKey) missing.push('NEXT_PUBLIC_SUPABASE_ANON_KEY');
    if (!serviceKey) missing.push('SUPABASE_SERVICE_ROLE_KEY');

    if (missing.length > 0) {
      tests.push({
        name: 'Variables d\'environnement Supabase',
        status: 'error',
        message: `Variables manquantes: ${missing.join(', ')}`,
        duration: Date.now() - start,
      });
    } else {
      tests.push({
        name: 'Variables d\'environnement Supabase',
        status: 'success',
        message: 'Toutes les variables Supabase sont configurées',
        duration: Date.now() - start,
      });
    }
  } catch (error) {
    tests.push({
      name: 'Variables d\'environnement Supabase',
      status: 'error',
      message: `Erreur: ${error instanceof Error ? error.message : 'Erreur inconnue'}`,
      duration: Date.now() - start,
    });
  }
}

async function testSupabaseConnection(tests: TestResult[]) {
  const start = Date.now();
  try {
    const { createSupabaseAdminClient } = await import('@/lib/supabase/admin');
    const supabase = createSupabaseAdminClient();

    // Test de connexion simple
    const { error } = await supabase.from('sites').select('id').limit(1);

    if (error) {
      tests.push({
        name: 'Connexion Supabase',
        status: 'error',
        message: `Erreur de connexion: ${error.message}`,
        details: error.code || undefined,
        duration: Date.now() - start,
      });
    } else {
      tests.push({
        name: 'Connexion Supabase',
        status: 'success',
        message: 'Connexion à Supabase réussie',
        duration: Date.now() - start,
      });
    }
  } catch (error) {
    tests.push({
      name: 'Connexion Supabase',
      status: 'error',
      message: `Erreur: ${error instanceof Error ? error.message : 'Erreur inconnue'}`,
      duration: Date.now() - start,
    });
  }
}

async function testDatabaseTables(tests: TestResult[]) {
  const start = Date.now();
  try {
    const { createSupabaseAdminClient } = await import('@/lib/supabase/admin');
    const supabase = createSupabaseAdminClient();

    const tables = ['sites', 'tasks', 'workers', 'reports'];
    const results: { table: string; accessible: boolean; error?: string }[] = [];

    for (const table of tables) {
      const { error } = await supabase.from(table).select('id').limit(1);
      results.push({
        table,
        accessible: !error,
        error: error?.message,
      });
    }

    const failed = results.filter(r => !r.accessible);
    if (failed.length > 0) {
      tests.push({
        name: 'Accès aux tables de base de données',
        status: 'error',
        message: `${failed.length} table(s) inaccessible(s): ${failed.map(f => f.table).join(', ')}`,
        details: failed.map(f => `${f.table}: ${f.error}`).join('; '),
        duration: Date.now() - start,
      });
    } else {
      tests.push({
        name: 'Accès aux tables de base de données',
        status: 'success',
        message: `Toutes les tables sont accessibles (${tables.length})`,
        duration: Date.now() - start,
      });
    }
  } catch (error) {
    tests.push({
      name: 'Accès aux tables de base de données',
      status: 'error',
      message: `Erreur: ${error instanceof Error ? error.message : 'Erreur inconnue'}`,
      duration: Date.now() - start,
    });
  }
}

async function testGeminiConfig(tests: TestResult[]) {
  const start = Date.now();
  try {
    const { isGeminiConfigured } = await import('@/lib/ai/gemini');
    const configured = await isGeminiConfigured();

    if (configured) {
      tests.push({
        name: 'Configuration Google Gemini',
        status: 'success',
        message: 'Google Gemini est configuré',
        duration: Date.now() - start,
      });
    } else {
      tests.push({
        name: 'Configuration Google Gemini',
        status: 'warning',
        message: 'Google Gemini n\'est pas configuré (fonctionnalité IA limitée)',
        details: 'La variable GOOGLE_GEMINI_API_KEY n\'est pas définie',
        duration: Date.now() - start,
      });
    }
  } catch (error) {
    tests.push({
      name: 'Configuration Google Gemini',
      status: 'error',
      message: `Erreur: ${error instanceof Error ? error.message : 'Erreur inconnue'}`,
      duration: Date.now() - start,
    });
  }
}

async function testGeminiAPI(tests: TestResult[]) {
  const start = Date.now();
  try {
    const { isGeminiConfigured, generateWithGemini } = await import('@/lib/ai/gemini');
    const configured = await isGeminiConfigured();

    if (!configured) {
      tests.push({
        name: 'Test API Google Gemini',
        status: 'warning',
        message: 'Test ignoré (Gemini non configuré)',
        duration: Date.now() - start,
      });
      return;
    }

    // Test simple de génération
    const result = await generateWithGemini('Réponds simplement "OK"', undefined, {
      maxOutputTokens: 10,
      temperature: 0.1,
    });

    if (result && result.trim().length > 0) {
      tests.push({
        name: 'Test API Google Gemini',
        status: 'success',
        message: 'L\'API Google Gemini répond correctement',
        duration: Date.now() - start,
      });
    } else {
      tests.push({
        name: 'Test API Google Gemini',
        status: 'error',
        message: 'L\'API Google Gemini ne retourne pas de réponse',
        duration: Date.now() - start,
      });
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Erreur inconnue';
    tests.push({
      name: 'Test API Google Gemini',
      status: 'error',
      message: `Erreur API: ${errorMessage}`,
      details: errorMessage.includes('QUOTA') ? 'Quota dépassé ou clé API invalide' : undefined,
      duration: Date.now() - start,
    });
  }
}

async function testAppConfig(tests: TestResult[]) {
  const start = Date.now();
  try {
    const baseUrl = process.env.NEXT_PUBLIC_APP_BASE_URL;

    if (!baseUrl) {
      tests.push({
        name: 'Configuration de l\'application',
        status: 'warning',
        message: 'NEXT_PUBLIC_APP_BASE_URL n\'est pas définie',
        details: 'Certaines fonctionnalités (emails, QR codes) peuvent ne pas fonctionner correctement',
        duration: Date.now() - start,
      });
    } else {
      tests.push({
        name: 'Configuration de l\'application',
        status: 'success',
        message: `URL de base configurée: ${baseUrl}`,
        duration: Date.now() - start,
      });
    }
  } catch (error) {
    tests.push({
      name: 'Configuration de l\'application',
      status: 'error',
      message: `Erreur: ${error instanceof Error ? error.message : 'Erreur inconnue'}`,
      duration: Date.now() - start,
    });
  }
}

async function testMLAPI(tests: TestResult[]) {
  const start = Date.now();
  try {
    const mlApiUrl = process.env.NEXT_PUBLIC_PREDICTION_API_URL || process.env.ML_API_URL || 'http://localhost:8000';

    try {
      const response = await fetch(`${mlApiUrl}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000), // 3 secondes max
      });

      if (response.ok) {
        tests.push({
          name: 'API Machine Learning',
          status: 'success',
          message: `L'API ML répond correctement (${mlApiUrl})`,
          duration: Date.now() - start,
        });
      } else {
        tests.push({
          name: 'API Machine Learning',
          status: 'warning',
          message: `L'API ML répond mais avec une erreur (${response.status})`,
          details: `URL: ${mlApiUrl}`,
          duration: Date.now() - start,
        });
      }
    } catch (error) {
      // Si c'est une erreur de timeout ou de connexion, c'est un warning (l'API peut ne pas être démarrée)
      if (error instanceof Error && (error.name === 'AbortError' || error.message.includes('fetch'))) {
        tests.push({
          name: 'API Machine Learning',
          status: 'warning',
          message: `L'API ML n'est pas accessible (${mlApiUrl})`,
          details: 'L\'API ML peut ne pas être démarrée. Ce n\'est pas critique si vous n\'utilisez pas les fonctionnalités ML.',
          duration: Date.now() - start,
        });
      } else {
        throw error;
      }
    }
  } catch (error) {
    tests.push({
      name: 'API Machine Learning',
      status: 'error',
      message: `Erreur: ${error instanceof Error ? error.message : 'Erreur inconnue'}`,
      duration: Date.now() - start,
    });
  }
}

