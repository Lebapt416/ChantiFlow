import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';

/**
 * API route pour vérifier la session worker
 */
export async function GET() {
  try {
    const cookieStore = await cookies();
    const workerSession = cookieStore.get('worker_session');

    if (!workerSession || !workerSession.value) {
      return NextResponse.json({ error: 'Pas de session' }, { status: 401 });
    }

    const session = JSON.parse(workerSession.value);

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

