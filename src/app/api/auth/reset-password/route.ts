import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

/**
 * Route API pour gérer le callback de réinitialisation de mot de passe
 * Supabase peut envoyer un code dans l'URL qui doit être échangé contre une session
 */
export async function GET(request: NextRequest) {
  const requestUrl = new URL(request.url);
  const code = requestUrl.searchParams.get('code');
  const type = requestUrl.searchParams.get('type');

  // Si c'est un lien de réinitialisation avec un code
  if (code && type === 'recovery') {
    const cookieStore = await cookies();
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseAnonKey) {
      return NextResponse.redirect(
        new URL('/login/reset-password?error=configuration', requestUrl.origin),
      );
    }

    const supabase = createServerClient(supabaseUrl, supabaseAnonKey, {
      cookies: {
        get(name: string) {
          return cookieStore.get(name)?.value;
        },
        set(name: string, value: string, options?: any) {
          try {
            cookieStore.set({
              name,
              value,
              maxAge: options?.maxAge || 60 * 60 * 24 * 365,
              path: options?.path || '/',
              sameSite: 'lax' as const,
              secure: process.env.NODE_ENV === 'production',
              httpOnly: false,
            });
          } catch {
            // Ignorer les erreurs
          }
        },
        remove(name: string, options?: any) {
          try {
            cookieStore.set({ name, value: '', ...options, maxAge: 0 });
          } catch {
            // Ignorer
          }
        },
      },
    });

    // Échanger le code contre une session
    const { error } = await supabase.auth.exchangeCodeForSession(code);

    if (error) {
      console.error('[reset-password route] Erreur échange code:', error);
      return NextResponse.redirect(
        new URL(`/login/reset-password?error=${encodeURIComponent(error.message)}`, requestUrl.origin),
      );
    }

    // Rediriger vers la page de réinitialisation avec la session établie
    return NextResponse.redirect(new URL('/login/reset-password', requestUrl.origin));
  }

  // Si pas de code, rediriger vers la page de réinitialisation
  return NextResponse.redirect(new URL('/login/reset-password', requestUrl.origin));
}

