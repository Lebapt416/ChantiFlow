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
 * AuthProvider - Fournisseur de contexte uniquement
 * 
 * Règles strictes :
 * 1. Ne JAMAIS appeler router.push() ou router.refresh()
 * 2. Ne JAMAIS forcer de redirection
 * 3. Rendre {children} IMMÉDIATEMENT (LCP optimisé)
 * 4. Mettre à jour l'état de session en arrière-plan
 * 5. Laisser le Middleware gérer toutes les redirections
 */
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const subscriptionRef = useRef<{ unsubscribe: () => void } | null>(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const supabase = createSupabaseBrowserClient();
    let isMounted = true;
    isMountedRef.current = true;

    // Rendre immédiatement pour optimiser le LCP
    setIsLoading(false);

    // Vérification initiale de session (non-bloquante)
    supabase.auth.getSession().then(({ data: { session }, error }) => {
      if (!isMounted || !isMountedRef.current) return;

      if (error && error.message.includes('Refresh Token')) {
        supabase.auth.signOut().catch(() => {});
        setUser(null);
        return;
      }

      if (session?.user) {
        setUser(session.user);
      } else {
        setUser(null);
      }
    }).catch(() => {
      // Ignorer silencieusement les erreurs
      if (isMounted && isMountedRef.current) {
        setUser(null);
      }
    });

    // Écouter les changements d'authentification (mise à jour d'état uniquement)
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      if (!isMounted || !isMountedRef.current) return;

      // Mettre à jour l'état local uniquement
      if (session?.user) {
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
