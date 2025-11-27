import { NextResponse } from 'next/server';
import { readWorkerSession } from '@/lib/worker-session';

/**
 * API route pour vérifier la session worker
 */
export async function GET() {
  try {
    const session = await readWorkerSession();
    if (!session?.workerId) {
      return NextResponse.json({ error: 'Pas de session' }, { status: 401 });
    }

    return NextResponse.json({
      workerId: session.workerId,
      siteId: session.siteId,
      name: session.name,
      email: session.email,
    });
  } catch (error) {
    console.error('Erreur lecture session worker:', error);
    return NextResponse.json({ error: 'Erreur lecture session' }, { status: 500 });
  }
}

