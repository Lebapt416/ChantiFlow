'use server';

import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export type AuthState = {
  error?: string;
  success?: string;
};

export async function signInAction(
  _prevState: AuthState,
  formData: FormData,
): Promise<AuthState> {
  const email = String(formData.get('email') ?? '').trim();
  const password = String(formData.get('password') ?? '');

  if (!email || !password) {
    return { error: 'Email et mot de passe sont requis.' };
  }

  const supabase = await createSupabaseServerClient({ allowCookieSetter: true });
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });

  if (error) {
    return { error: error.message };
  }

  // Rediriger vers /analytics si c'est le compte admin analytics (par ID ou email)
  const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
  if (data.user?.id === authorizedUserId || data.user?.email === 'bcb83@icloud.com') {
    redirect('/analytics');
  }

  redirect('/home');
}

export async function signUpAction(
  _prevState: AuthState,
  formData: FormData,
): Promise<AuthState> {
  const email = String(formData.get('email') ?? '').trim();
  const password = String(formData.get('password') ?? '');
  const name = String(formData.get('name') ?? '').trim();

  if (!email || !password) {
    return { error: 'Email et mot de passe sont requis.' };
  }

  if (password.length < 6) {
    return { error: 'Le mot de passe doit contenir au moins 6 caractères.' };
  }

  const supabase = await createSupabaseServerClient({ allowCookieSetter: true });
  
  // Créer l'utilisateur
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: {
        name: name || undefined,
        plan: 'basic',
      },
      emailRedirectTo: `${process.env.NEXT_PUBLIC_APP_BASE_URL || 'http://localhost:3000'}/auth/callback?next=/home`,
    },
  });

  if (error) {
    return { error: error.message };
  }

  if (data.user) {
    // Rediriger vers /analytics si c'est le compte admin analytics
    const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
    if (data.user.id === authorizedUserId || data.user.email === 'bcb83@icloud.com') {
      redirect('/analytics');
    }
    
    // Si l'email n'est pas confirmé, afficher un message
    if (!data.session) {
      return { 
        success: 'Compte créé ! Vérifiez votre email pour confirmer votre compte avant de vous connecter.' 
      };
    }
    
    redirect('/home');
  }

  return { error: 'Une erreur est survenue lors de la création du compte.' };
}

