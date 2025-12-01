'use client';

import { useEffect, useState } from 'react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { useRouter, usePathname } from 'next/navigation';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    // S'assurer que nous sommes côté client
    if (typeof window === 'undefined') {
      // eslint-disable-next-line react-hooks/exhaustive-deps
      setTimeout(() => setIsChecking(false), 0);
      return;
    }

    // Vérifier si on est en mode PWA (standalone)
    const isPWA = window.matchMedia('(display-mode: standalone)').matches;

    // Restaurer la session au chargement de l'app (important pour PWA)
    const supabase = createSupabaseBrowserClient();

    // Vérifier et restaurer la session au chargement
    supabase.auth.getSession().then(({ data: { session }, error }) => {
      // Gérer les erreurs de refresh token invalide silencieusement
      if (error && error.message.includes('Refresh Token')) {
        console.warn('Token de rafraîchissement invalide, nettoyage de la session');
        supabase.auth.signOut().catch(() => {
          // Ignorer les erreurs de déconnexion
        });
        setIsChecking(false);
        return;
      }
      
      setIsChecking(false);
      
      if (session) {
        console.log('✅ Session restaurée automatiquement:', session.user.email);
        
        const currentPath = window.location.pathname;
        
        // En PWA uniquement : rediriger automatiquement si on est sur login ou landing
        if (isPWA && (currentPath === '/login' || currentPath === '/landing')) {
          // Vérifier si c'est le compte analytics
          const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
          if (session.user.id === authorizedUserId || session.user.email === 'bcb83@icloud.com') {
            router.push('/analytics');
          } else {
            router.push('/home');
          }
        }
        // Sur le site web : ne rien faire, laisser l'utilisateur sur la landing page
      } else {
        console.log('❌ Aucune session trouvée');
        
        // Si pas de session et qu'on est sur une page protégée, rediriger vers login
        const currentPath = window.location.pathname;
        const protectedPaths = ['/home', '/dashboard', '/sites', '/planning', '/reports', '/qr', '/team', '/analytics'];
        const isProtectedPath = protectedPaths.some(path => currentPath.startsWith(path));
        
        if (isProtectedPath) {
          router.push('/login');
        }
      }
    }).catch((error) => {
      console.error('Erreur lors de la vérification de session:', error);
      // Nettoyer la session en cas d'erreur
      if (error.message?.includes('Refresh Token')) {
        supabase.auth.signOut().catch(() => {
          // Ignorer les erreurs de déconnexion
        });
      }
      setIsChecking(false);
    });

    // Écouter les changements d'authentification
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      const isPWA = window.matchMedia('(display-mode: standalone)').matches;
      
      if (event === 'SIGNED_IN' && session) {
        console.log('✅ Utilisateur connecté:', session.user.email);
        setIsChecking(false);
        
        // Rediriger automatiquement après connexion
        const currentPath = window.location.pathname;
        const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
        
        if (session.user.id === authorizedUserId || session.user.email === 'bcb83@icloud.com') {
          router.push('/analytics');
        } else if (currentPath === '/login') {
          // Après connexion depuis la page login, toujours rediriger vers /home
          router.push('/home');
        } else if (isPWA && currentPath === '/landing') {
          // En PWA uniquement : rediriger depuis landing vers home
          router.push('/home');
        } else {
          router.refresh();
        }
      } else if (event === 'SIGNED_OUT') {
        console.log('❌ Utilisateur déconnecté');
        setIsChecking(false);
        const currentPath = window.location.pathname;
        // Rediriger vers login seulement si on est sur une page protégée
        const protectedPaths = ['/home', '/dashboard', '/sites', '/planning', '/reports', '/qr', '/team', '/analytics'];
        if (protectedPaths.some(path => currentPath.startsWith(path))) {
          router.push('/login');
        }
      } else if (event === 'TOKEN_REFRESHED' && session) {
        console.log('🔄 Token rafraîchi');
        // Rafraîchir silencieusement pour mettre à jour les cookies serveur
        router.refresh();
      }
    });

    return () => {
      subscription.unsubscribe();
    };
  }, [router]);

  // Ne pas afficher de loader côté serveur pour éviter les problèmes d'hydratation
  // Le loader ne s'affichera que côté client après le premier rendu
  if (isChecking) {
    // Utiliser usePathname pour éviter les problèmes d'hydratation
    const currentPath = pathname;
    if (currentPath === '/login' || currentPath === '/landing' || currentPath === '/') {
      return (
        <div className="min-h-screen bg-white dark:bg-zinc-950">
          <div className="fixed inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-black border-r-transparent dark:border-white dark:border-r-transparent"></div>
              <p className="mt-4 text-zinc-600 dark:text-zinc-400">Chargement...</p>
            </div>
          </div>
        </div>
      );
    }
  }

  return <>{children}</>;
}

