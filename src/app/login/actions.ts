'use server';

import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export type AuthState = {
  error?: string;
  success?: string;
};

const SUPABASE_PLACEHOLDER_HOSTS = ['example.supabase.co'];
const SUPABASE_PLACEHOLDER_KEYS = ['example', 'your-anon-key'];

const SUPABASE_DISABLED_MESSAGE =
  "Erreur : l'authentification est désactivée sur cet environnement (configuration Supabase manquante).";
const SUPABASE_UNREACHABLE_MESSAGE =
  "Erreur : impossible de joindre le service d'authentification pour le moment.";

function isSupabaseConfigured() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

  if (!url || !anonKey) {
    return false;
  }

  if (SUPABASE_PLACEHOLDER_HOSTS.some((host) => url.includes(host))) {
    return false;
  }

  if (SUPABASE_PLACEHOLDER_KEYS.some((fragment) => anonKey.includes(fragment))) {
    return false;
  }

  return true;
}

export async function signInAction(
  _prevState: AuthState,
  formData: FormData,
): Promise<AuthState> {
  const email = String(formData.get('email') ?? '').trim();
  const password = String(formData.get('password') ?? '');

  if (!email || !password) {
    return { error: 'Email et mot de passe sont requis.' };
  }

  if (!isSupabaseConfigured()) {
    console.warn('Supabase non configuré: retour erreur mock pour les tests E2E.');
    return { error: SUPABASE_DISABLED_MESSAGE };
  }

  try {
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
  } catch (error) {
    console.error('Erreur signInAction (Supabase indisponible)', error);
    return { error: SUPABASE_UNREACHABLE_MESSAGE };
  }

  // Satisfait TypeScript : redirect ou retour
  return { error: SUPABASE_UNREACHABLE_MESSAGE };
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

  if (!isSupabaseConfigured()) {
    console.warn('Supabase non configuré: signUp impossible sur cet environnement.');
    return { error: SUPABASE_DISABLED_MESSAGE };
  }

  try {
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
  } catch (error) {
    console.error('Erreur signUpAction (Supabase indisponible)', error);
    return { error: SUPABASE_UNREACHABLE_MESSAGE };
  }
}

export async function resetPasswordRequestAction(
  _prevState: AuthState,
  formData: FormData,
): Promise<AuthState> {
  const email = String(formData.get('email') ?? '').trim();

  if (!email) {
    return { error: 'Email requis.' };
  }

  if (!isSupabaseConfigured()) {
    return { error: SUPABASE_DISABLED_MESSAGE };
  }

  try {
    const supabase = await createSupabaseServerClient({ allowCookieSetter: true });
    const baseUrl = process.env.NEXT_PUBLIC_APP_BASE_URL || 'http://localhost:3000';
    const redirectTo = `${baseUrl}/login/reset-password`;

    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo,
    });

    if (error) {
      return { error: error.message };
    }

    // Toujours retourner un succès pour ne pas révéler si l'email existe ou non
    return {
      success: 'Si cet email existe, un lien de réinitialisation a été envoyé.',
    };
  } catch (error) {
    console.error('Erreur resetPasswordRequestAction', error);
    return { error: SUPABASE_UNREACHABLE_MESSAGE };
  }
}

export async function resetPasswordAction(
  _prevState: AuthState,
  formData: FormData,
): Promise<AuthState> {
  const password = String(formData.get('password') ?? '');
  const confirmPassword = String(formData.get('confirmPassword') ?? '');

  if (!password || !confirmPassword) {
    return { error: 'Les deux champs de mot de passe sont requis.' };
  }

  if (password.length < 6) {
    return { error: 'Le mot de passe doit contenir au moins 6 caractères.' };
  }

  if (password !== confirmPassword) {
    return { error: 'Les mots de passe ne correspondent pas.' };
  }

  if (!isSupabaseConfigured()) {
    return { error: SUPABASE_DISABLED_MESSAGE };
  }

  try {
    const supabase = await createSupabaseServerClient({ allowCookieSetter: true });
    
    // Vérifier que l'utilisateur est authentifié via le token de réinitialisation
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    
    if (userError || !user) {
      return { error: 'Lien de réinitialisation invalide ou expiré. Demandez un nouveau lien.' };
    }

    // Mettre à jour uniquement le mot de passe (les autres données utilisateur restent intactes)
    const { error } = await supabase.auth.updateUser({
      password,
    });

    if (error) {
      return { error: error.message };
    }

    // Rediriger vers la page de connexion avec un message de succès
    redirect('/login?reset=success');
  } catch (error) {
    console.error('Erreur resetPasswordAction', error);
    return { error: SUPABASE_UNREACHABLE_MESSAGE };
  }

  // Satisfait TypeScript
  return { error: SUPABASE_UNREACHABLE_MESSAGE };
}

