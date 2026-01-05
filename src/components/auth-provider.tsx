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
    let isInitialCheck = true; // Flag pour ignorer le premier événement INITIAL_SESSION

    // Vérification initiale de session (une seule fois)
    supabase.auth.getSession().then(({ data: { session }, error }) => {
      if (!isMounted) return;

      if (error && error.message.includes('Refresh Token')) {
        supabase.auth.signOut().catch(() => {});
        return;
      }

      if (session) {
        previousUserIdRef.current = session.user.id;
        previousPathRef.current = pathname || window.location.pathname;
      }
    }).catch(() => {
      // Ignorer les erreurs silencieusement
    });

    // Écouter les changements d'authentification avec protection contre les boucles
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      if (!isMounted) return;

      // IGNORER l'événement INITIAL_SESSION qui se déclenche au chargement
      // C'est la cause principale des rafraîchissements infinis
      if (event === 'INITIAL_SESSION') {
        isInitialCheck = false;
        if (session) {
          previousUserIdRef.current = session.user.id;
          previousPathRef.current = pathname || window.location.pathname;
        }
        return; // Ne rien faire sur INITIAL_SESSION
      }

      const currentUserId = session?.user?.id || null;
      const currentPath = pathname || window.location.pathname;
      const isPWA = window.matchMedia('(display-mode: standalone)').matches;
      const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';

      // Protection contre les boucles : ne rediriger que si l'état a vraiment changé
      const userIdChanged = previousUserIdRef.current !== currentUserId;
      const pathChanged = previousPathRef.current !== currentPath;

      // Ne rediriger QUE si l'utilisateur vient de se connecter (SIGNED_IN) et que c'est un vrai changement
      if (event === 'SIGNED_IN' && session && userIdChanged && !isInitialCheck) {
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
        // Ne pas appeler router.refresh() ici pour éviter les boucles
      } else if (event === 'SIGNED_OUT' && userIdChanged && !isInitialCheck) {
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
      } else if (event === 'TOKEN_REFRESHED' && session) {
        // Ne PAS appeler router.refresh() ici pour éviter les boucles infinies
        // Le token est rafraîchi automatiquement, pas besoin de recharger la page
        previousUserIdRef.current = session.user.id;
      }
    });

    subscriptionRef.current = subscription;

    return () => {
      isMounted = false;
      if (subscriptionRef.current) {
        subscriptionRef.current.unsubscribe();
      }
    };
  }, [router, pathname]);

  // Ne jamais bloquer le rendu initial - toujours afficher le contenu
  // La vérification de session se fait de manière asynchrone en arrière-plan
  return <>{children}</>;
}

