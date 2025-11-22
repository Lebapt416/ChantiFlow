'use server';

import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export type AuthState = {
  error?: string;
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

  // Rediriger vers /analytics si c'est le compte admin analytics
  if (data.user?.email === 'bcb83@icloud.com') {
    redirect('/analytics');
  }

  redirect('/home');
}

