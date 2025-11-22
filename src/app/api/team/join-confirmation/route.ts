import { NextRequest, NextResponse } from 'next/server';
import { sendTeamJoinConfirmationEmail } from '@/lib/email';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';

export async function POST(request: NextRequest) {
  try {
    const { workerEmail, workerName, userId } = await request.json();

    console.log('📧 API join-confirmation appelée avec:', { workerEmail, workerName, userId });

    if (!workerEmail || !workerName || !userId) {
      console.error('❌ Paramètres manquants:', { workerEmail, workerName, userId });
      return NextResponse.json(
        { error: 'Paramètres manquants' },
        { status: 400 }
      );
    }

    // Récupérer l'email du manager
    const admin = createSupabaseAdminClient();
    const { data: manager, error: managerError } = await admin.auth.admin.getUserById(userId);
    
    if (managerError) {
      console.error('❌ Erreur récupération manager:', managerError);
    }
    
    const managerEmail = manager?.user?.email;
    console.log('📧 Email manager récupéré:', managerEmail);

    // Envoyer l'email de confirmation
    console.log('📧 Appel sendTeamJoinConfirmationEmail avec:', { workerEmail, workerName, managerName: managerEmail });
    const result = await sendTeamJoinConfirmationEmail({
      workerEmail,
      workerName,
      managerName: managerEmail,
    });

    console.log('📧 Résultat envoi email:', result);

    if (!result.success) {
      console.error('❌ Erreur envoi email:', result.error);
      return NextResponse.json(
        { error: result.error, success: false },
        { status: 500 }
      );
    }

    console.log('✅ Email envoyé avec succès');
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('❌ Exception API join-confirmation:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Erreur serveur', success: false },
      { status: 500 }
    );
  }
}

