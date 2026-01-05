'use client';

import { createContext, useContext, useEffect, useRef, useState } from 'react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import type { User } from '@supabase/supabase-js';

/**
 * Contexte d'authentification - Diffusion uniquement, PAS de redirection
 * Les redirections sont gérées par le Middleware côté serveur
 */
type AuthContextType = {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
};

const AuthContext = createContext<AuthContextType>({
  user: null,
  isLoading: true,
  isAuthenticated: false,
});

export function useAuth() {
  return useContext(AuthContext);
}

/**
 * Nettoyer le localStorage en préservant les clés importantes
 */
function clearLocalStorageSafely() {
  try {
    const themeKey = 'chantiflow-theme';
    const theme = localStorage.getItem(themeKey);
    
    // Vider complètement localStorage
    localStorage.clear();
    
    // Restaurer uniquement le thème si présent
    if (theme) {
      localStorage.setItem(themeKey, theme);
    }
  } catch (error) {
    console.error('[AuthProvider] Erreur lors du nettoyage localStorage:', error);
  }
}

/**
 * AuthProvider - Fournisseur de contexte avec disjoncteur de session
 * 
 * Règles strictes :
 * 1. Ne JAMAIS appeler router.push() ou router.refresh()
 * 2. Ne JAMAIS forcer de redirection
 * 3. Rendre {children} IMMÉDIATEMENT (LCP optimisé)
 * 4. Disjoncteur de session : nettoyage automatique si erreur ou timeout > 5s
 * 5. Laisser le Middleware gérer toutes les redirections
 */
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const subscriptionRef = useRef<{ unsubscribe: () => void } | null>(null);
  const isMountedRef = useRef(true);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const hasCleanedRef = useRef(false);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const supabase = createSupabaseBrowserClient();
    let isMounted = true;
    isMountedRef.current = true;

    // Rendre immédiatement pour optimiser le LCP
    setIsLoading(false);

    // DISJONCTEUR DE SESSION : Timeout de 5 secondes
    // MAIS seulement si on est sur une page publique (pas pendant la connexion)
    const isPublicPage = window.location.pathname === '/' || 
                         window.location.pathname === '/login' ||
                         window.location.pathname.startsWith('/contact') ||
                         window.location.pathname.startsWith('/team/join');
    
    // Ne pas déclencher le disjoncteur sur les pages publiques (pour permettre la connexion)
    if (!isPublicPage) {
      timeoutRef.current = setTimeout(() => {
        if (isMounted && isMountedRef.current && !hasCleanedRef.current) {
          console.warn('[AuthProvider] ⚠️ Timeout de session (5s) - Nettoyage automatique');
          hasCleanedRef.current = true;
          
          // Nettoyer la session et le localStorage
          supabase.auth.signOut().catch(() => {});
          clearLocalStorageSafely();
          setUser(null);
        }
      }, 5000);
    }

    // Vérification initiale de session (non-bloquante)
    const sessionPromise = supabase.auth.getSession().then(({ data: { session }, error }) => {
      // Annuler le timeout si la vérification réussit
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      if (!isMounted || !isMountedRef.current) return;

      // DISJONCTEUR : Si erreur d'authentification, nettoyer
      // MAIS seulement si on n'est pas sur la page de login (pour permettre la connexion)
      if (error) {
        const errorMessage = error.message || '';
        const isOnLoginPage = window.location.pathname === '/login' || 
                              window.location.pathname.startsWith('/login');
        
        // Ne pas nettoyer sur la page de login (l'utilisateur est en train de se connecter)
        if (!isOnLoginPage && !hasCleanedRef.current) {
          console.warn('[AuthProvider] ⚠️ Erreur d\'authentification détectée:', errorMessage);
          hasCleanedRef.current = true;
          
          // Nettoyer la session et le localStorage
          supabase.auth.signOut().catch(() => {});
          clearLocalStorageSafely();
          setUser(null);
        }
        
        // Mettre à jour l'état même sur la page de login
        setUser(null);
        return;
      }

      // Session valide
      if (session?.user) {
        setUser(session.user);
      } else {
        setUser(null);
      }
    }).catch((error) => {
      // Annuler le timeout en cas d'erreur
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      console.error('[AuthProvider] ❌ Erreur lors de la vérification de session:', error);
      
      if (isMounted && isMountedRef.current && !hasCleanedRef.current) {
        hasCleanedRef.current = true;
        
        // Nettoyer en cas d'erreur
        supabase.auth.signOut().catch(() => {});
        clearLocalStorageSafely();
        setUser(null);
      }
    });

    // Écouter les changements d'authentification (mise à jour d'état uniquement)
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      if (!isMounted || !isMountedRef.current) return;

      // Annuler le timeout si on reçoit un événement valide
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      // DISJONCTEUR : Si erreur dans l'événement, nettoyer
      if (event === 'SIGNED_OUT' || event === 'USER_UPDATED') {
        // Mettre à jour l'état local uniquement
        if (session?.user) {
          setUser(session.user);
        } else {
          setUser(null);
        }
      } else if (session?.user) {
        setUser(session.user);
      } else {
        setUser(null);
      }

      // Ne JAMAIS déclencher de redirection ici
      // Le Middleware gère toutes les redirections côté serveur
    });

    subscriptionRef.current = subscription;

    return () => {
      isMounted = false;
      isMountedRef.current = false;
      
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      
      if (subscriptionRef.current) {
        subscriptionRef.current.unsubscribe();
        subscriptionRef.current = null;
      }
    };
  }, []); // AUCUNE dépendance - exécution unique

  // Rendre immédiatement pour optimiser le LCP
  // Ne jamais bloquer le rendu initial
  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}
