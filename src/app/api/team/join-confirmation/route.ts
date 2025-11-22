import { NextRequest, NextResponse } from 'next/server';
import { sendTeamJoinConfirmationEmail } from '@/lib/email';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';

export async function POST(request: NextRequest) {
  try {
    const { workerEmail, workerName, userId } = await request.json();

    if (!workerEmail || !workerName || !userId) {
      return NextResponse.json(
        { error: 'Paramètres manquants' },
        { status: 400 }
      );
    }

    // Récupérer l'email du manager
    const admin = createSupabaseAdminClient();
    const { data: manager } = await admin.auth.admin.getUserById(userId);
    const managerEmail = manager?.user?.email;

    // Envoyer l'email de confirmation
    const result = await sendTeamJoinConfirmationEmail({
      workerEmail,
      workerName,
      managerName: managerEmail,
    });

    if (!result.success) {
      console.error('Erreur envoi email:', result.error);
      return NextResponse.json(
        { error: result.error },
        { status: 500 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Erreur API join-confirmation:', error);
    return NextResponse.json(
      { error: 'Erreur serveur' },
      { status: 500 }
    );
  }
}

