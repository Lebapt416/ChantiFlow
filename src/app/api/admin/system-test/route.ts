import { NextRequest, NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { isAdmin } from '@/lib/admin';
import { runSystemTests } from '@/lib/system-test';

/**
 * Route API pour exécuter les tests système (admin uniquement)
 */
export async function GET(request: NextRequest) {
  try {
    // Vérifier l'authentification
    const supabase = await createSupabaseServerClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user || !user.email) {
      return NextResponse.json(
        { error: 'Non authentifié' },
        { status: 401 }
      );
    }

    // Vérifier que l'utilisateur est admin
    if (!isAdmin(user.email)) {
      return NextResponse.json(
        { error: 'Accès refusé. Administrateur requis.' },
        { status: 403 }
      );
    }

    // Exécuter les tests
    const report = await runSystemTests();

    return NextResponse.json(report);
  } catch (error) {
    console.error('Erreur lors des tests système:', error);
    return NextResponse.json(
      {
        error: 'Erreur lors de l\'exécution des tests',
        details: error instanceof Error ? error.message : 'Erreur inconnue',
      },
      { status: 500 }
    );
  }
}

