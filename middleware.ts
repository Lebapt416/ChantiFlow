import { createServerClient } from '@supabase/ssr';
import { NextResponse, type NextRequest } from 'next/server';

/**
 * Middleware - Gestion centralisée de toutes les redirections
 * 
 * Règles strictes :
 * 1. Toutes les redirections sont gérées ici (côté serveur)
 * 2. Protection contre les boucles de redirection
 * 3. Synchronisation correcte des cookies Supabase
 * 4. Suppression active des cookies malformés
 * 5. Redirections uniquement si le path est différent
 */
export async function middleware(request: NextRequest) {
  const response = NextResponse.next();
  
  // Liste des cookies Supabase à vérifier
  const supabaseCookieNames = [
    'sb-access-token',
    'sb-refresh-token',
    'supabase-auth-token',
  ];

  // Vérifier et nettoyer les cookies malformés
  const cookiesToRemove: string[] = [];
  for (const cookieName of supabaseCookieNames) {
    const cookie = request.cookies.get(cookieName);
    if (cookie) {
      const value = cookie.value;
      // Détecter les cookies malformés (vide, trop court, ou format invalide)
      if (!value || value.length < 10 || value === 'undefined' || value === 'null') {
        cookiesToRemove.push(cookieName);
      }
    }
  }

  // Supprimer activement les cookies malformés
  if (cookiesToRemove.length > 0) {
    console.warn('[Middleware] 🧹 Suppression de cookies malformés:', cookiesToRemove);
    for (const cookieName of cookiesToRemove) {
      response.cookies.delete(cookieName);
      // Supprimer aussi dans la requête pour éviter la propagation
      request.cookies.delete(cookieName);
    }
  }

  // Créer un client Supabase pour le middleware avec synchronisation des cookies
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value, options }) => {
            // Synchroniser les cookies dans la requête et la réponse
            request.cookies.set({ name, value, ...options });
            response.cookies.set({ name, value, ...options });
          });
        },
      },
    }
  );

  // Vérifier la session avec timeout strict pour éviter les timeouts Edge
  const authTimeoutMs = 1500;
  const authResult = await Promise.race([
    supabase.auth.getUser(),
    new Promise<{
      data: { user: null };
      error: { message: string };
    }>((resolve) =>
      setTimeout(
        () =>
          resolve({
            data: { user: null },
            error: { message: 'Auth timeout in middleware' },
          }),
        authTimeoutMs,
      ),
    ),
  ]);

  const {
    data: { user },
    error: authError,
  } = authResult;

  const pathname = request.nextUrl.pathname;
  const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
  const isAuthorized = user?.id === authorizedUserId || user?.email === 'bcb83@icloud.com';

  // Chemins publics (accessibles sans authentification)
  const publicPaths = ['/', '/login', '/contact', '/team/join', '/team/join/success'];
  const isPublicPath = publicPaths.some(path => pathname === path || pathname.startsWith(path));

  // Chemins protégés (nécessitent authentification)
  const protectedPaths = [
    '/home',
    '/dashboard',
    '/sites',
    '/planning',
    '/reports',
    '/qr',
    '/team',
    '/analytics',
    '/account',
    '/worker',
  ];
  const isProtectedPath = protectedPaths.some(path => pathname.startsWith(path));

  // Protection contre les boucles de redirection
  // Vérifier si on est déjà en train de rediriger vers la même page
  const referer = request.headers.get('referer');
  let isRedirectLoop = false;
  if (referer) {
    try {
      isRedirectLoop = new URL(referer).pathname === pathname;
    } catch {
      isRedirectLoop = false;
    }
  }

  // Cas 1 : Utilisateur non authentifié tentant d'accéder à une route protégée
  if (!user && isProtectedPath && !isPublicPath) {
    // Protection contre boucle : si on est déjà sur /login, ne pas rediriger
    if (pathname !== '/login' && !isRedirectLoop) {
      const loginUrl = new URL('/login', request.url);
      return NextResponse.redirect(loginUrl);
    }
  }

  // Cas 2 : Utilisateur authentifié tentant d'accéder à /login ou /
  if (user && (pathname === '/login' || pathname === '/')) {
    // Protection contre boucle : vérifier qu'on ne redirige pas vers la même page
    if (!isRedirectLoop) {
      if (isAuthorized) {
        // Admin → /analytics
        const analyticsUrl = new URL('/analytics', request.url);
        return NextResponse.redirect(analyticsUrl);
      } else {
        // Utilisateur normal → /home
        const homeUrl = new URL('/home', request.url);
        return NextResponse.redirect(homeUrl);
      }
    }
  }

  // Cas 3 : Gestion des erreurs d'authentification
  // Ne nettoyer les cookies QUE si l'erreur est vraiment critique (pas juste une absence de session)
  if (authError && authError.message && 
      (authError.message.includes('JWT') || authError.message.includes('expired') || authError.message.includes('invalid')) &&
      isProtectedPath && !isPublicPath) {
    // Token invalide ou expiré → nettoyer les cookies et rediriger vers login
    console.warn('[Middleware] 🧹 Erreur d\'authentification critique - Nettoyage des cookies');
    
    // Supprimer tous les cookies Supabase
    for (const cookieName of supabaseCookieNames) {
      response.cookies.delete(cookieName);
      request.cookies.delete(cookieName);
    }
    
    if (pathname !== '/login' && !isRedirectLoop) {
      const loginUrl = new URL('/login', request.url);
      return NextResponse.redirect(loginUrl);
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
     * - api routes (gestion séparée)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|api|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
};
