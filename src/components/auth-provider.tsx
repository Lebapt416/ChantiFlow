'use client';

import { useEffect, useState, useRef } from 'react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { useRouter, usePathname } from 'next/navigation';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [isChecking, setIsChecking] = useState(true);
  
  // Refs pour tracker l'état précédent et éviter les boucles infinies
  const previousUserIdRef = useRef<string | null>(null);
  const previousPathRef = useRef<string | null>(null);
  const hasRedirectedRef = useRef(false);
  const subscriptionRef = useRef<{ unsubscribe: () => void } | null>(null);

  // Ne jamais bloquer le rendu initial
  useEffect(() => {
    setIsChecking(false);
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const supabase = createSupabaseBrowserClient();
    let isMounted = true;
    let hasInitialized = false; // Flag pour s'assurer qu'on ne s'abonne qu'une fois

    // Vérification initiale de session (une seule fois)
    supabase.auth.getSession().then(({ data: { session }, error }) => {
      if (!isMounted) return;

      if (error && error.message.includes('Refresh Token')) {
        supabase.auth.signOut().catch(() => {});
        return;
      }

      if (session) {
        previousUserIdRef.current = session.user.id;
        previousPathRef.current = window.location.pathname;
      }
      hasInitialized = true;
    }).catch(() => {
      hasInitialized = true;
    });

    // Écouter les changements d'authentification avec protection contre les boucles
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      if (!isMounted || !hasInitialized) return;

      // IGNORER complètement INITIAL_SESSION et TOKEN_REFRESHED
      // Ce sont les causes principales des rafraîchissements infinis
      if (event === 'INITIAL_SESSION' || event === 'TOKEN_REFRESHED') {
        if (session) {
          previousUserIdRef.current = session.user.id;
          previousPathRef.current = window.location.pathname;
        }
        return; // Ne rien faire sur ces événements
      }

      const currentUserId = session?.user?.id || null;
      const currentPath = window.location.pathname;
      const isPWA = window.matchMedia('(display-mode: standalone)').matches;
      const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';

      // Protection contre les boucles : ne rediriger que si l'état a vraiment changé
      const userIdChanged = previousUserIdRef.current !== currentUserId;

      // Ne rediriger QUE sur SIGNED_IN ou SIGNED_OUT réels (pas INITIAL_SESSION)
      if (event === 'SIGNED_IN' && session && userIdChanged) {
        previousUserIdRef.current = currentUserId;
        previousPathRef.current = currentPath;
        hasRedirectedRef.current = false;

        // Rediriger uniquement depuis /login ou /landing (PWA)
        if (currentPath === '/login' || currentPath.startsWith('/login')) {
          if (session.user.id === authorizedUserId || session.user.email === 'bcb83@icloud.com') {
            if (!hasRedirectedRef.current) {
              hasRedirectedRef.current = true;
              router.push('/analytics');
            }
          } else {
            if (!hasRedirectedRef.current) {
              hasRedirectedRef.current = true;
              router.push('/home');
            }
          }
        } else if (isPWA && currentPath === '/landing') {
          if (!hasRedirectedRef.current) {
            hasRedirectedRef.current = true;
            router.push('/home');
          }
        }
        // Ne JAMAIS appeler router.refresh() ici
      } else if (event === 'SIGNED_OUT' && userIdChanged) {
        previousUserIdRef.current = null;
        previousPathRef.current = currentPath;
        hasRedirectedRef.current = false;

        const publicPaths = ['/team/join', '/team/join/success', '/login', '/', '/contact'];
        const isPublicPath = publicPaths.some(path => currentPath.startsWith(path));

        if (!isPublicPath) {
          const protectedPaths = ['/home', '/dashboard', '/sites', '/planning', '/reports', '/qr', '/team', '/analytics'];
          if (protectedPaths.some(path => currentPath.startsWith(path))) {
            if (!hasRedirectedRef.current) {
              hasRedirectedRef.current = true;
              router.push('/login');
            }
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
  }, [router]); // Retirer pathname des dépendances pour éviter les re-renders

  // Ne jamais bloquer le rendu initial - toujours afficher le contenu
  // La vérification de session se fait de manière asynchrone en arrière-plan
  return <>{children}</>;
}

