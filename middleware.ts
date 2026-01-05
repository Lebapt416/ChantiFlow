import { createServerClient } from '@supabase/ssr';
import { NextResponse, type NextRequest } from 'next/server';

/**
 * Middleware pour synchroniser la logique d'authentification serveur/client
 * Évite les conflits entre redirections serveur et client
 */
export async function middleware(request: NextRequest) {
  const response = NextResponse.next();
  
  // Créer un client Supabase pour le middleware
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name: string) {
          return request.cookies.get(name)?.value;
        },
        set(name: string, value: string, options: any) {
          request.cookies.set({ name, value, ...options });
          response.cookies.set({ name, value, ...options });
        },
        remove(name: string, options: any) {
          request.cookies.set({ name, value: '', ...options });
          response.cookies.set({ name, value: '', ...options });
        },
      },
    }
  );

  // Vérifier la session (sans bloquer)
  const {
    data: { user },
  } = await supabase.auth.getUser();

  const pathname = request.nextUrl.pathname;
  const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
  const isAuthorized = user?.id === authorizedUserId || user?.email === 'bcb83@icloud.com';

  // Chemins publics (pas de redirection)
  const publicPaths = ['/', '/login', '/contact', '/team/join', '/team/join/success'];
  const isPublicPath = publicPaths.some(path => pathname === path || pathname.startsWith(path));

  // Chemins protégés (nécessitent authentification)
  const protectedPaths = ['/home', '/dashboard', '/sites', '/planning', '/reports', '/qr', '/team', '/analytics', '/account'];
  const isProtectedPath = protectedPaths.some(path => pathname.startsWith(path));

  // Redirection uniquement si nécessaire et si le path est différent
  if (!user && isProtectedPath) {
    // Pas de session et chemin protégé → rediriger vers login
    if (pathname !== '/login') {
      const loginUrl = new URL('/login', request.url);
      return NextResponse.redirect(loginUrl);
    }
  } else if (user && pathname === '/login') {
    // Session active et sur /login → rediriger selon le rôle
    if (isAuthorized) {
      const analyticsUrl = new URL('/analytics', request.url);
      return NextResponse.redirect(analyticsUrl);
    } else {
      const homeUrl = new URL('/home', request.url);
      return NextResponse.redirect(homeUrl);
    }
  }

  // Laisser passer toutes les autres requêtes
  return response;
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
};

