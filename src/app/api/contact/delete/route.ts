import { NextRequest, NextResponse } from 'next/server';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';

export async function DELETE(request: NextRequest) {
  try {
    // Vérifier que l'utilisateur est admin
    const { createSupabaseServerClient } = await import('@/lib/supabase/server');
    const { isAdmin } = await import('@/lib/admin');
    const supabase = await createSupabaseServerClient();
    const { data: { user } } = await supabase.auth.getUser();

    if (!user || !isAdmin(user.email)) {
      return NextResponse.json(
        { error: 'Non autorisé.' },
        { status: 403 }
      );
    }

    const { searchParams } = new URL(request.url);
    const messageId = searchParams.get('id');

    if (!messageId) {
      return NextResponse.json(
        { error: 'ID du message requis.' },
        { status: 400 }
      );
    }

    const adminClient = createSupabaseAdminClient();
    
    const { error } = await adminClient
      .from('contact_messages')
      .delete()
      .eq('id', messageId);

    if (error) {
      console.error('Erreur suppression message:', error);
      return NextResponse.json(
        { error: 'Erreur lors de la suppression du message.' },
        { status: 500 }
      );
    }

    return NextResponse.json(
      { success: true, message: 'Message supprimé avec succès' },
      { status: 200 }
    );
  } catch (error) {
    console.error('Erreur API contact delete:', error);
    return NextResponse.json(
      { error: 'Une erreur est survenue. Veuillez réessayer.' },
      { status: 500 }
    );
  }
}

