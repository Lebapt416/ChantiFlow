import { NextRequest, NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email, password } = body;

    if (!email || !password) {
      return NextResponse.json(
        { error: 'Email et mot de passe sont requis.' },
        { status: 400 }
      );
    }

    const supabase = await createSupabaseServerClient({ allowCookieSetter: true });
    
    const { data, error } = await supabase.auth.signInWithPassword({
      email: email.trim(),
      password,
    });

    if (error) {
      console.error('[API SignIn] Erreur Supabase:', error);
      
      if (error.message.includes('Invalid login credentials') || error.message.includes('Invalid credentials')) {
        return NextResponse.json(
          { error: 'Email ou mot de passe incorrect.' },
          { status: 401 }
        );
      }
      
      if (error.message.includes('Email not confirmed')) {
        return NextResponse.json(
          { error: 'Votre email n\'est pas confirmé. Vérifiez votre boîte de réception.' },
          { status: 401 }
        );
      }
      
      return NextResponse.json(
        { error: error.message || 'Erreur lors de la connexion.' },
        { status: 500 }
      );
    }

    if (!data.user || !data.session) {
      return NextResponse.json(
        { error: 'Erreur lors de la création de la session.' },
        { status: 500 }
      );
    }

    // Déterminer la redirection : admin → /analytics, sinon → /home
    const { isAdmin } = await import('@/lib/admin');
    const redirectTo = isAdmin(data.user.email) ? '/analytics' : '/home';

    return NextResponse.json({
      success: true,
      redirectTo,
      user: {
        id: data.user.id,
        email: data.user.email,
      },
    });
  } catch (error) {
    console.error('[API SignIn] Erreur inattendue:', error);
    return NextResponse.json(
      { error: 'Erreur lors de la connexion. Veuillez réessayer.' },
      { status: 500 }
    );
  }
}

