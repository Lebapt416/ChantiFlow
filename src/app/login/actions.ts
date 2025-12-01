'use server';

import { redirect } from 'next/navigation';
import { headers } from 'next/headers';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export type AuthState = {
  error?: string;
  success?: string;
  redirectTo?: string;
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
    console.warn('[isSupabaseConfigured] Variables manquantes:', {
      hasUrl: !!url,
      hasKey: !!anonKey,
    });
    return false;
  }

  if (SUPABASE_PLACEHOLDER_HOSTS.some((host) => url.includes(host))) {
    console.warn('[isSupabaseConfigured] URL placeholder détectée');
    return false;
  }

  if (SUPABASE_PLACEHOLDER_KEYS.some((fragment) => anonKey.includes(fragment))) {
    console.warn('[isSupabaseConfigured] Clé placeholder détectée');
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
    console.warn('[signInAction] Supabase non configuré');
    return { error: SUPABASE_DISABLED_MESSAGE };
  }

  try {
    console.log('[signInAction] Tentative de connexion pour:', email);
    console.log('[signInAction] Variables env:', {
      hasUrl: !!process.env.NEXT_PUBLIC_SUPABASE_URL,
      hasKey: !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      urlLength: process.env.NEXT_PUBLIC_SUPABASE_URL?.length || 0,
    });
    
    let supabase;
    try {
      supabase = await createSupabaseServerClient({ allowCookieSetter: true });
      console.log('[signInAction] Client Supabase créé avec succès');
    } catch (createError) {
      console.error('[signInAction] Erreur création client Supabase:', createError);
      if (createError instanceof Error && createError.message.includes('Supabase est mal configuré')) {
        return { error: SUPABASE_DISABLED_MESSAGE };
      }
      throw createError;
    }
    
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    
    console.log('[signInAction] Réponse Supabase:', { 
      hasUser: !!data?.user, 
      hasSession: !!data?.session,
      error: error?.message 
    });

    if (error) {
      console.error('[signInAction] Erreur Supabase auth:', error);
      // Messages d'erreur plus spécifiques
      if (error.message.includes('Invalid login credentials') || error.message.includes('Invalid credentials')) {
        return { error: 'Email ou mot de passe incorrect.' };
      }
      if (error.message.includes('Email not confirmed')) {
        return { error: 'Votre email n\'est pas confirmé. Vérifiez votre boîte de réception.' };
      }
      return { error: error.message || 'Erreur lors de la connexion. Veuillez réessayer.' };
    }

    if (!data.user) {
      console.error('[signInAction] Pas d\'utilisateur retourné');
      return { error: 'Erreur lors de la connexion. Veuillez réessayer.' };
    }

    if (!data.session) {
      console.error('[signInAction] Pas de session créée');
      return { error: 'Erreur lors de la création de la session. Veuillez réessayer.' };
    }

    console.log('[signInAction] Connexion réussie, session créée');

    // Retourner l'URL de redirection au lieu de rediriger directement
    // (redirect() ne fonctionne pas toujours avec useActionState)
    const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
    const redirectTo = (data.user.id === authorizedUserId || data.user.email === 'bcb83@icloud.com') 
      ? '/analytics' 
      : '/home';

    return { success: 'Connexion réussie', redirectTo };
  } catch (error) {
    // Gérer spécifiquement les erreurs de configuration
    if (error instanceof Error) {
      console.error('[signInAction] Erreur détaillée:', {
        message: error.message,
        stack: error.stack,
        name: error.name,
      });
      
      if (error.message.includes('Supabase est mal configuré')) {
        return { error: SUPABASE_DISABLED_MESSAGE };
      }
      
      // Erreurs réseau
      if (error.message.includes('fetch') || error.message.includes('network') || error.message.includes('ECONNREFUSED')) {
        return { error: 'Impossible de se connecter à Supabase. Vérifiez votre connexion internet.' };
      }
    }
    
    console.error('[signInAction] Erreur inattendue:', error);
    // Message générique mais plus utile
    return { error: 'Erreur lors de la connexion. Vérifiez vos identifiants ou réessayez plus tard.' };
  }
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
    
    // Détecter l'URL actuelle depuis les headers de la requête
    // Cela permet de fonctionner en localhost ET en production
    let baseUrl = process.env.NEXT_PUBLIC_APP_BASE_URL || 'http://localhost:3000';
    
    try {
      const headersList = await headers();
      const host = headersList.get('host');
      const protocol = headersList.get('x-forwarded-proto') || (process.env.NODE_ENV === 'production' ? 'https' : 'http');
      
      if (host) {
        // Utiliser l'URL de la requête actuelle
        baseUrl = `${protocol}://${host}`;
        console.log('[resetPasswordRequestAction] URL détectée depuis headers:', baseUrl);
      }
    } catch (headerError) {
      // Si on ne peut pas accéder aux headers (par exemple dans certaines Server Actions),
      // utiliser la variable d'environnement
      console.log('[resetPasswordRequestAction] Utilisation de NEXT_PUBLIC_APP_BASE_URL:', baseUrl);
    }
    
    // IMPORTANT: Cette URL doit être autorisée dans Supabase Dashboard > Authentication > URL Configuration > Redirect URLs
    // Utiliser la route API pour gérer le code de réinitialisation
    const redirectTo = `${baseUrl}/api/auth/reset-password`;

    console.log('[resetPasswordRequestAction] Envoi email de réinitialisation pour:', email);
    console.log('[resetPasswordRequestAction] URL de redirection:', redirectTo);

    const { data, error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo,
      // Options supplémentaires pour améliorer la compatibilité
      captchaToken: undefined, // Pas de captcha pour l'instant
    });

    if (error) {
      console.error('[resetPasswordRequestAction] Erreur Supabase:', error);
      // Messages d'erreur plus explicites
      if (error.message.includes('rate limit')) {
        return { error: 'Trop de tentatives. Veuillez patienter quelques minutes avant de réessayer.' };
      }
      if (error.message.includes('email')) {
        return { error: 'Format d\'email invalide.' };
      }
      return { error: `Erreur: ${error.message}` };
    }

    console.log('[resetPasswordRequestAction] Email envoyé avec succès');

    // Toujours retourner un succès pour ne pas révéler si l'email existe ou non
    return {
      success: 'Si cet email existe, un lien de réinitialisation a été envoyé. Vérifiez votre boîte de réception (et les spams).',
    };
  } catch (error) {
    console.error('[resetPasswordRequestAction] Erreur inattendue:', error);
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
    // La session devrait avoir été établie côté client via le hash de l'URL
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    
    if (userError) {
      console.error('[resetPasswordAction] Erreur getUser:', userError);
      return { error: 'Lien de réinitialisation invalide ou expiré. Demandez un nouveau lien.' };
    }
    
    if (!user) {
      console.error('[resetPasswordAction] Pas d\'utilisateur trouvé');
      return { error: 'Lien de réinitialisation invalide ou expiré. Le lien peut avoir expiré (valable 1 heure).' };
    }

    console.log('[resetPasswordAction] Mise à jour du mot de passe pour:', user.email);

    // Mettre à jour uniquement le mot de passe (les autres données utilisateur restent intactes)
    const { error: updateError } = await supabase.auth.updateUser({
      password,
    });

    if (updateError) {
      console.error('[resetPasswordAction] Erreur updateUser:', updateError);
      return { error: updateError.message || 'Erreur lors de la mise à jour du mot de passe.' };
    }

    console.log('[resetPasswordAction] Mot de passe mis à jour avec succès');

    // Déconnecter l'utilisateur après la réinitialisation pour qu'il se reconnecte avec le nouveau mot de passe
    await supabase.auth.signOut();

    // Rediriger vers la page de connexion avec un message de succès
    redirect('/login?reset=success');
  } catch (error) {
    console.error('[resetPasswordAction] Erreur inattendue:', error);
    return { error: SUPABASE_UNREACHABLE_MESSAGE };
  }

  // Satisfait TypeScript
  return { error: SUPABASE_UNREACHABLE_MESSAGE };
}

