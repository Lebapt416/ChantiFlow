import { NextRequest, NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email, password, name } = body;

    if (!email || !password) {
      return NextResponse.json(
        { error: 'Email et mot de passe sont requis.' },
        { status: 400 }
      );
    }

    if (password.length < 6) {
      return NextResponse.json(
        { error: 'Le mot de passe doit contenir au moins 6 caractères.' },
        { status: 400 }
      );
    }

    const supabase = await createSupabaseServerClient({ allowCookieSetter: true });
    
    const { data, error } = await supabase.auth.signUp({
      email: email.trim(),
      password,
      options: {
        data: {
          name: name || undefined,
          plan: 'basic',
        },
        emailRedirectTo: `${process.env.NEXT_PUBLIC_APP_BASE_URL || 'https://www.chantiflow.com'}/auth/callback?next=/home`,
      },
    });

    if (error) {
      console.error('[API SignUp] Erreur Supabase:', error);
      return NextResponse.json(
        { error: error.message || 'Erreur lors de la création du compte.' },
        { status: 500 }
      );
    }

    if (!data.user) {
      return NextResponse.json(
        { error: 'Erreur lors de la création du compte.' },
        { status: 500 }
      );
    }

    // Si l'email n'est pas confirmé, retourner un message
    if (!data.session) {
      return NextResponse.json({
        success: 'Compte créé ! Vérifiez votre email pour confirmer votre compte avant de vous connecter.',
      });
    }

    // Si la session est créée, rediriger
    const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
    const redirectTo = (data.user.id === authorizedUserId || data.user.email === 'bcb83@icloud.com') 
      ? '/analytics' 
      : '/home';

    return NextResponse.json({
      success: true,
      redirectTo,
      user: {
        id: data.user.id,
        email: data.user.email,
      },
    });
  } catch (error) {
    console.error('[API SignUp] Erreur inattendue:', error);
    return NextResponse.json(
      { error: 'Erreur lors de la création du compte. Veuillez réessayer.' },
      { status: 500 }
    );
  }
}

