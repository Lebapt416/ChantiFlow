'use client';

import { useEffect, useState } from 'react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { useRouter, usePathname } from 'next/navigation';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    // Restaurer la session au chargement de l'app (important pour PWA)
    const supabase = createSupabaseBrowserClient();

    // Vérifier et restaurer la session au chargement
    supabase.auth.getSession().then(({ data: { session } }) => {
      setIsChecking(false);
      
      if (session) {
        console.log('✅ Session restaurée automatiquement pour PWA:', session.user.email);
        
        // Rediriger automatiquement si on est sur login ou landing
        if (pathname === '/login' || pathname === '/landing') {
          // Vérifier si c'est le compte analytics
          const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
          if (session.user.id === authorizedUserId || session.user.email === 'bcb83@icloud.com') {
            router.push('/analytics');
          } else {
            router.push('/home');
          }
        }
      } else {
        console.log('❌ Aucune session trouvée');
        
        // Si pas de session et qu'on est sur une page protégée, rediriger vers login
        const protectedPaths = ['/home', '/dashboard', '/sites', '/planning', '/reports', '/qr', '/team', '/analytics'];
        const isProtectedPath = protectedPaths.some(path => pathname?.startsWith(path));
        
        if (isProtectedPath) {
          router.push('/login');
        } else if (pathname === '/') {
          // Si on est sur la racine sans session, aller à landing
          router.push('/landing');
        }
      }
    });

    // Écouter les changements d'authentification pour la PWA
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      if (event === 'SIGNED_IN' && session) {
        console.log('✅ Utilisateur connecté:', session.user.email);
        setIsChecking(false);
        
        // Rediriger automatiquement après connexion
        const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
        if (session.user.id === authorizedUserId || session.user.email === 'bcb83@icloud.com') {
          router.push('/analytics');
        } else if (pathname === '/login' || pathname === '/landing') {
          router.push('/home');
        } else {
          router.refresh();
        }
      } else if (event === 'SIGNED_OUT') {
        console.log('❌ Utilisateur déconnecté');
        setIsChecking(false);
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

  // Afficher un loader pendant la vérification de session
  if (isChecking && (pathname === '/login' || pathname === '/landing' || pathname === '/')) {
    return (
      <div className="fixed inset-0 flex items-center justify-center bg-white dark:bg-zinc-950">
        <div className="text-center">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-black border-r-transparent dark:border-white dark:border-r-transparent"></div>
          <p className="mt-4 text-zinc-600 dark:text-zinc-400">Chargement...</p>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}

