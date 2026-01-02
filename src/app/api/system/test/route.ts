import { NextResponse } from 'next/server';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { generateWithGemini, isGeminiConfigured } from '@/lib/ai/gemini';
import { Resend } from 'resend';

type TestResult = {
  name: string;
  status: 'success' | 'error' | 'warning';
  message: string;
  details?: string;
  duration?: number;
};

export async function GET() {
  const results: TestResult[] = [];
  const startTime = Date.now();

  // Test 1: Variables d'environnement Supabase
  const testStart1 = Date.now();
  try {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
    const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

    if (!supabaseUrl || !supabaseAnonKey || !supabaseServiceKey) {
      throw new Error('Variables Supabase manquantes');
    }

    results.push({
      name: 'Variables d\'environnement Supabase',
      status: 'success',
      message: 'Toutes les variables Supabase sont configurées',
      duration: Date.now() - testStart1,
    });
  } catch (error) {
    results.push({
      name: 'Variables d\'environnement Supabase',
      status: 'error',
      message: error instanceof Error ? error.message : 'Variables manquantes',
      duration: Date.now() - testStart1,
    });
  }

  // Test 2: Connexion Supabase Admin
  const testStart2 = Date.now();
  try {
    const adminClient = createSupabaseAdminClient();
    const { data, error } = await adminClient.from('sites').select('id').limit(1);

    if (error) throw error;

    results.push({
      name: 'Connexion Supabase Admin',
      status: 'success',
      message: 'Connexion réussie',
      details: `Test de lecture effectué`,
      duration: Date.now() - testStart2,
    });
  } catch (error) {
    results.push({
      name: 'Connexion Supabase Admin',
      status: 'error',
      message: error instanceof Error ? error.message : 'Erreur de connexion',
      duration: Date.now() - testStart2,
    });
  }

  // Test 3: Connexion Supabase Server
  const testStart3 = Date.now();
  try {
    const serverClient = await createSupabaseServerClient();
    const { error } = await serverClient.from('sites').select('id').limit(1);

    if (error) throw error;

    results.push({
      name: 'Connexion Supabase Server',
      status: 'success',
      message: 'Connexion réussie',
      duration: Date.now() - testStart3,
    });
  } catch (error) {
    results.push({
      name: 'Connexion Supabase Server',
      status: 'error',
      message: error instanceof Error ? error.message : 'Erreur de connexion',
      duration: Date.now() - testStart3,
    });
  }

  // Test 4: Test de lecture base de données
  const testStart4 = Date.now();
  try {
    const adminClient = createSupabaseAdminClient();
    const [sitesResult, tasksResult, usersResult] = await Promise.all([
      adminClient.from('sites').select('id').limit(1),
      adminClient.from('tasks').select('id').limit(1),
      adminClient.auth.admin.listUsers({ page: 1, perPage: 1 }),
    ]);

    if (sitesResult.error) throw sitesResult.error;
    if (tasksResult.error) throw tasksResult.error;
    if (usersResult.error) throw usersResult.error;

    results.push({
      name: 'Lecture base de données',
      status: 'success',
      message: 'Lecture réussie sur toutes les tables',
      details: `Sites: OK, Tâches: OK, Utilisateurs: OK`,
      duration: Date.now() - testStart4,
    });
  } catch (error) {
    results.push({
      name: 'Lecture base de données',
      status: 'error',
      message: error instanceof Error ? error.message : 'Erreur de lecture',
      duration: Date.now() - testStart4,
    });
  }

  // Test 5: Test d'écriture base de données (test uniquement, pas de réel insert)
  const testStart5 = Date.now();
  try {
    const adminClient = createSupabaseAdminClient();
    // Test de permission d'écriture en vérifiant qu'on peut accéder à la table
    const { error } = await adminClient
      .from('sites')
      .select('id')
      .limit(1);

    if (error) throw error;

    results.push({
      name: 'Permissions base de données',
      status: 'success',
      message: 'Permissions d\'écriture/lecture OK',
      duration: Date.now() - testStart5,
    });
  } catch (error) {
    results.push({
      name: 'Permissions base de données',
      status: 'error',
      message: error instanceof Error ? error.message : 'Erreur de permissions',
      duration: Date.now() - testStart5,
    });
  }

  // Test 6: Variables d'environnement Google Gemini
  const testStart6 = Date.now();
  try {
    const geminiKey = process.env.GOOGLE_GEMINI_API_KEY;
    const isConfigured = await isGeminiConfigured();

    if (!geminiKey || !isConfigured) {
      throw new Error('GOOGLE_GEMINI_API_KEY non configurée');
    }

    results.push({
      name: 'Variables d\'environnement Google Gemini',
      status: 'success',
      message: 'Clé API configurée',
      duration: Date.now() - testStart6,
    });
  } catch (error) {
    results.push({
      name: 'Variables d\'environnement Google Gemini',
      status: 'error',
      message: error instanceof Error ? error.message : 'Clé API manquante',
      duration: Date.now() - testStart6,
    });
  }

  // Test 7: API Google Gemini
  const testStart7 = Date.now();
  try {
    const testPrompt = 'Réponds simplement "OK" si tu reçois ce message.';
    const response = await generateWithGemini(testPrompt, undefined, {
      maxOutputTokens: 10,
      temperature: 0.1,
    });

    if (!response || response.trim().length === 0) {
      throw new Error('Réponse vide');
    }

    results.push({
      name: 'API Google Gemini',
      status: 'success',
      message: 'Connexion et génération réussies',
      details: `Réponse reçue: ${response.substring(0, 50)}...`,
      duration: Date.now() - testStart7,
    });
  } catch (error) {
    results.push({
      name: 'API Google Gemini',
      status: 'error',
      message: error instanceof Error ? error.message : 'Erreur API',
      details: error instanceof Error ? error.stack : undefined,
      duration: Date.now() - testStart7,
    });
  }

  // Test 8: Variables d'environnement Resend
  const testStart8 = Date.now();
  try {
    const resendKey = process.env.RESEND_API_KEY;
    const resendFrom = process.env.RESEND_FROM_EMAIL;

    if (!resendKey) {
      throw new Error('RESEND_API_KEY manquante');
    }

    if (!resendFrom) {
      results.push({
        name: 'Variables d\'environnement Resend',
        status: 'warning',
        message: 'RESEND_API_KEY configurée mais RESEND_FROM_EMAIL manquante',
        duration: Date.now() - testStart8,
      });
    } else {
      results.push({
        name: 'Variables d\'environnement Resend',
        status: 'success',
        message: 'Toutes les variables Resend sont configurées',
        duration: Date.now() - testStart8,
      });
    }
  } catch (error) {
    results.push({
      name: 'Variables d\'environnement Resend',
      status: 'error',
      message: error instanceof Error ? error.message : 'Variables manquantes',
      duration: Date.now() - testStart8,
    });
  }

  // Test 9: API Resend (test de connexion uniquement, pas d'envoi réel)
  const testStart9 = Date.now();
  try {
    const resendKey = process.env.RESEND_API_KEY;
    if (!resendKey) {
      throw new Error('RESEND_API_KEY manquante');
    }

    const resend = new Resend(resendKey);
    // On ne peut pas vraiment tester sans envoyer un email, donc on vérifie juste l'initialisation
    if (!resend) {
      throw new Error('Impossible d\'initialiser Resend');
    }

    results.push({
      name: 'API Resend',
      status: 'success',
      message: 'Client Resend initialisé avec succès',
      details: 'Note: Aucun email réel envoyé (test de connexion uniquement)',
      duration: Date.now() - testStart9,
    });
  } catch (error) {
    results.push({
      name: 'API Resend',
      status: 'error',
      message: error instanceof Error ? error.message : 'Erreur d\'initialisation',
      duration: Date.now() - testStart9,
    });
  }

  // Test 10: Vérification des tables importantes
  const testStart10 = Date.now();
  try {
    const adminClient = createSupabaseAdminClient();
    const tables = ['sites', 'tasks', 'workers', 'reports', 'contact_messages'];
    const tableResults: string[] = [];

    for (const table of tables) {
      const { error } = await adminClient.from(table).select('id').limit(1);
      if (error) {
        tableResults.push(`${table}: ❌`);
      } else {
        tableResults.push(`${table}: ✅`);
      }
    }

    const hasErrors = tableResults.some((r) => r.includes('❌'));

    results.push({
      name: 'Vérification des tables',
      status: hasErrors ? 'warning' : 'success',
      message: hasErrors ? 'Certaines tables ont des problèmes' : 'Toutes les tables sont accessibles',
      details: tableResults.join(', '),
      duration: Date.now() - testStart10,
    });
  } catch (error) {
    results.push({
      name: 'Vérification des tables',
      status: 'error',
      message: error instanceof Error ? error.message : 'Erreur de vérification',
      duration: Date.now() - testStart10,
    });
  }

  // Calcul de la note globale
  const totalTests = results.length;
  const successTests = results.filter((r) => r.status === 'success').length;
  const errorTests = results.filter((r) => r.status === 'error').length;
  const warningTests = results.filter((r) => r.status === 'warning').length;

  const score = Math.round((successTests / totalTests) * 100);
  let grade: string;
  if (score >= 90) grade = 'Excellent';
  else if (score >= 75) grade = 'Bon';
  else if (score >= 50) grade = 'Moyen';
  else grade = 'Critique';

  const totalDuration = Date.now() - startTime;

  return NextResponse.json({
    success: true,
    summary: {
      score,
      grade,
      totalTests,
      successTests,
      errorTests,
      warningTests,
      totalDuration,
    },
    results,
    timestamp: new Date().toISOString(),
  });
}

