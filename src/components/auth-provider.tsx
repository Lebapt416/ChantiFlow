'use client';

import { useEffect } from 'react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { useRouter, usePathname } from 'next/navigation';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    // Restaurer la session au chargement de l'app (important pour PWA)
    const supabase = createSupabaseBrowserClient();

    // Vérifier et restaurer la session au chargement
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) {
        console.log('✅ Session restaurée pour PWA:', session.user.email);
        // Si on est sur la page de login et qu'on a une session, rediriger
        if (pathname === '/login') {
          router.push('/home');
        }
      } else {
        // Si pas de session et qu'on est sur une page protégée, rediriger vers login
        const protectedPaths = ['/home', '/dashboard', '/sites', '/planning', '/reports', '/qr', '/team', '/analytics'];
        if (protectedPaths.some(path => pathname?.startsWith(path))) {
          router.push('/login');
        }
      }
    });

    // Écouter les changements d'authentification pour la PWA
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      if (event === 'SIGNED_IN' && session) {
        console.log('✅ Utilisateur connecté:', session.user.email);
        // Rafraîchir la page pour mettre à jour l'état serveur
        if (pathname === '/login') {
          router.push('/home');
        } else {
          router.refresh();
        }
      } else if (event === 'SIGNED_OUT') {
        console.log('❌ Utilisateur déconnecté');
        router.push('/login');
      } else if (event === 'TOKEN_REFRESHED' && session) {
        console.log('🔄 Token rafraîchi pour PWA');
        // Rafraîchir silencieusement pour mettre à jour les cookies serveur
        router.refresh();
      }
    });

    return () => {
      subscription.unsubscribe();
    };
  }, [router, pathname]);

  return <>{children}</>;
}

