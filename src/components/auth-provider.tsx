'use client';

import { useEffect } from 'react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { useRouter } from 'next/navigation';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();

  useEffect(() => {
    // Restaurer la session au chargement de l'app (important pour PWA)
    const supabase = createSupabaseBrowserClient();

    // Vérifier et restaurer la session
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) {
        console.log('Session restaurée pour PWA:', session.user.email);
      }
    });

    // Écouter les changements d'authentification
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === 'SIGNED_IN' && session) {
        console.log('Utilisateur connecté:', session.user.email);
        // Rafraîchir la page pour mettre à jour l'état serveur
        router.refresh();
      } else if (event === 'SIGNED_OUT') {
        console.log('Utilisateur déconnecté');
        router.push('/login');
      } else if (event === 'TOKEN_REFRESHED' && session) {
        console.log('Token rafraîchi pour PWA');
        // Rafraîchir silencieusement pour mettre à jour les cookies
        router.refresh();
      }
    });

    return () => {
      subscription.unsubscribe();
    };
  }, [router]);

  return <>{children}</>;
}

