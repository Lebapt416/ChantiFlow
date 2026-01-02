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

    // Ne pas bloquer le rendu initial - vérifier la session de manière asynchrone
    setIsChecking(false);

    // Vérifier si on est en mode PWA (standalone)
    const isPWA = window.matchMedia('(display-mode: standalone)').matches;

    // Restaurer la session au chargement de l'app (important pour PWA)
    // Fait de manière asynchrone pour ne pas bloquer le FCP
    const supabase = createSupabaseBrowserClient();

    // Vérifier et restaurer la session au chargement (non-bloquant)
    supabase.auth.getSession().then(({ data: { session }, error }) => {
      // Gérer les erreurs de refresh token invalide silencieusement
      if (error && error.message.includes('Refresh Token')) {
        console.warn('Token de rafraîchissement invalide, nettoyage de la session');
        supabase.auth.signOut().catch(() => {
          // Ignorer les erreurs de déconnexion
        });
        return;
      }
      
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
        // Pages publiques qui ne nécessitent pas d'authentification
        const publicPaths = ['/team/join', '/team/join/success', '/login', '/', '/contact'];
        const isPublicPath = publicPaths.some(path => currentPath.startsWith(path));
        
        if (!isPublicPath) {
          const protectedPaths = ['/home', '/dashboard', '/sites', '/planning', '/reports', '/qr', '/team', '/analytics'];
          const isProtectedPath = protectedPaths.some(path => currentPath.startsWith(path));
          
          if (isProtectedPath) {
            router.push('/login');
          }
        }
      }
    }).catch((error) => {
      console.error('Erreur lors de la vérification de session:', error);
      // Nettoyer la session en cas d'erreur
      if (error?.message?.includes('Refresh Token')) {
        supabase.auth.signOut().catch(() => {
          // Ignorer les erreurs de déconnexion
        });
      }
      // Ne pas bloquer même en cas d'erreur - le contenu s'affiche quand même
    });

    // Écouter les changements d'authentification
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      const isPWA = window.matchMedia('(display-mode: standalone)').matches;
      
      if (event === 'SIGNED_IN' && session) {
        console.log('✅ Utilisateur connecté via onAuthStateChange:', session.user.email);
        setIsChecking(false);
        
        // Rediriger automatiquement après connexion
        const currentPath = window.location.pathname;
        const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
        
        // Rediriger immédiatement (pas besoin d'attendre, les cookies sont déjà définis)
        if (session.user.id === authorizedUserId || session.user.email === 'bcb83@icloud.com') {
          router.push('/analytics');
        } else if (currentPath === '/login' || currentPath.startsWith('/login')) {
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
        // Pages publiques qui ne nécessitent pas d'authentification
        const publicPaths = ['/team/join', '/team/join/success', '/login', '/', '/contact'];
        const isPublicPath = publicPaths.some(path => currentPath.startsWith(path));
        
        if (!isPublicPath) {
          // Rediriger vers login seulement si on est sur une page protégée
          const protectedPaths = ['/home', '/dashboard', '/sites', '/planning', '/reports', '/qr', '/team', '/analytics'];
          if (protectedPaths.some(path => currentPath.startsWith(path))) {
            router.push('/login');
          }
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

  // Ne jamais bloquer le rendu initial - toujours afficher le contenu
  // La vérification de session se fait de manière asynchrone en arrière-plan
  return <>{children}</>;
}

