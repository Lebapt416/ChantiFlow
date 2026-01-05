'use client';

import { useEffect, useRef } from 'react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { useRouter, usePathname } from 'next/navigation';

/**
 * AuthProvider optimisé - Zéro re-render, Zéro boucle infinie
 * 
 * Règles strictes :
 * 1. Ne jamais appeler router.refresh()
 * 2. Ne rediriger QUE si pathname est différent de la destination
 * 3. Ignorer complètement INITIAL_SESSION et TOKEN_REFRESHED
 * 4. Vérification immédiate mais non-bloquante
 */
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  
  // Refs pour tracker l'état et éviter les boucles
  const previousUserIdRef = useRef<string | null>(null);
  const previousPathRef = useRef<string | null>(null);
  const hasRedirectedRef = useRef(false);
  const subscriptionRef = useRef<{ unsubscribe: () => void } | null>(null);
  const isInitializedRef = useRef(false);
  const routerRef = useRef(router);
  const pathnameRef = useRef(pathname);

  // Synchroniser les refs sans déclencher de re-render
  useEffect(() => {
    routerRef.current = router;
    pathnameRef.current = pathname;
  }, [router, pathname]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const supabase = createSupabaseBrowserClient();
    let isMounted = true;

    // Vérification initiale IMMÉDIATE mais non-bloquante
    supabase.auth.getSession().then(({ data: { session }, error }) => {
      if (!isMounted) return;

      if (error && error.message.includes('Refresh Token')) {
        supabase.auth.signOut().catch(() => {});
        return;
      }

      if (session) {
        previousUserIdRef.current = session.user.id;
        previousPathRef.current = pathnameRef.current;
      }
      isInitializedRef.current = true;
    }).catch(() => {
      isInitializedRef.current = true;
    });

    // Écouter les changements d'authentification
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      if (!isMounted || !isInitializedRef.current) return;

      // IGNORER complètement ces événements (causes de boucles)
      if (event === 'INITIAL_SESSION' || event === 'TOKEN_REFRESHED') {
        if (session) {
          previousUserIdRef.current = session.user.id;
          previousPathRef.current = pathnameRef.current;
        }
        return; // Ne rien faire
      }

      const currentUserId = session?.user?.id || null;
      const currentPath = pathnameRef.current;
      const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
      const isAuthorized = currentUserId === authorizedUserId || session?.user?.email === 'bcb83@icloud.com';

      // Protection stricte : ne rediriger QUE si l'utilisateur a changé
      const userIdChanged = previousUserIdRef.current !== currentUserId;

      // SIGNED_IN : Rediriger uniquement depuis /login et si pathname différent
      if (event === 'SIGNED_IN' && session && userIdChanged) {
        previousUserIdRef.current = currentUserId;
        previousPathRef.current = currentPath;
        hasRedirectedRef.current = false;

        // Condition STRICTE : pathname doit être /login ET destination différente
        if (currentPath === '/login' || currentPath.startsWith('/login')) {
          if (isAuthorized) {
            // Rediriger vers /analytics SEULEMENT si on n'y est pas déjà
            if (currentPath !== '/analytics') {
              hasRedirectedRef.current = true;
              routerRef.current.push('/analytics');
            }
          } else {
            // Rediriger vers /home SEULEMENT si on n'y est pas déjà
            if (currentPath !== '/home') {
              hasRedirectedRef.current = true;
              routerRef.current.push('/home');
            }
          }
        }
        // Ne JAMAIS appeler router.refresh() ici
      } 
      // SIGNED_OUT : Rediriger uniquement depuis chemins protégés
      else if (event === 'SIGNED_OUT' && userIdChanged) {
        previousUserIdRef.current = null;
        previousPathRef.current = currentPath;
        hasRedirectedRef.current = false;

        const publicPaths = ['/team/join', '/team/join/success', '/login', '/', '/contact'];
        const isPublicPath = publicPaths.some(path => currentPath === path || currentPath.startsWith(path));

        if (!isPublicPath) {
          const protectedPaths = ['/home', '/dashboard', '/sites', '/planning', '/reports', '/qr', '/team', '/analytics', '/account'];
          const isProtectedPath = protectedPaths.some(path => currentPath.startsWith(path));
          
          // Rediriger vers /login SEULEMENT si on n'y est pas déjà
          if (isProtectedPath && currentPath !== '/login') {
            hasRedirectedRef.current = true;
            routerRef.current.push('/login');
          }
        }
      }
    });

    subscriptionRef.current = subscription;

    return () => {
      isMounted = false;
      if (subscriptionRef.current) {
        subscriptionRef.current.unsubscribe();
        subscriptionRef.current = null;
      }
    };
  }, []); // AUCUNE dépendance - exécution unique

  // Ne jamais bloquer le rendu - toujours afficher immédiatement
  return <>{children}</>;
}
