import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function GET(request: NextRequest) {
  const requestUrl = new URL(request.url);
  const code = requestUrl.searchParams.get('code');
  const next = requestUrl.searchParams.get('next') || '/dashboard';

  if (code) {
    const cookieStore = await cookies();
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseAnonKey) {
      return NextResponse.redirect(
        new URL('/login?error=configuration', requestUrl.origin),
      );
    }

    const supabase = createServerClient(supabaseUrl, supabaseAnonKey, {
      cookies: {
        get(name: string) {
          return cookieStore.get(name)?.value;
        },
        set(name: string, value: string, options: any) {
          try {
            cookieStore.set({
              name,
              value,
              ...options,
              maxAge: options?.maxAge || 60 * 60 * 24 * 365, // 1 an par défaut
              sameSite: 'lax' as const,
              secure: process.env.NODE_ENV === 'production',
              httpOnly: false, // Permettre l'accès depuis JavaScript pour la PWA
            });
          } catch {
            // Ignorer les erreurs de mutation de cookies
          }
        },
        remove(name: string, options: any) {
          try {
            cookieStore.set({
              name,
              value: '',
              ...options,
              maxAge: 0,
            });
          } catch {
            // Ignorer les erreurs de suppression de cookies
          }
        },
      },
    });

    // Échanger le code contre une session
    const { error } = await supabase.auth.exchangeCodeForSession(code);

    if (error) {
      console.error('Erreur échange code pour session:', error);
      return NextResponse.redirect(
        new URL(`/login?error=${encodeURIComponent(error.message)}`, requestUrl.origin),
      );
    }

    // Rediriger vers la page demandée ou le dashboard
    return NextResponse.redirect(new URL(next, requestUrl.origin));
  }

  // Si pas de code, rediriger vers la page de connexion
  return NextResponse.redirect(new URL('/login', requestUrl.origin));
}

