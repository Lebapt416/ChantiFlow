'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { ResetPasswordForm } from './reset-password-form';

export function ResetPasswordHandler() {
  const [isValidating, setIsValidating] = useState(true);
  const [isValidToken, setIsValidToken] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    async function validateToken() {
      try {
        const supabase = createSupabaseBrowserClient();
        
        // Vérifier s'il y a une erreur dans l'URL (venant de la route API)
        const errorParam = searchParams.get('error');
        if (errorParam) {
          setIsValidToken(false);
          setError(decodeURIComponent(errorParam));
          setIsValidating(false);
          return;
        }
        
        // Attendre un peu pour que le hash/code soit traité par Supabase
        // Essayer plusieurs fois avec des délais progressifs
        let attempts = 0;
        const maxAttempts = 3;
        let user = null;

        while (attempts < maxAttempts && !user) {
          await new Promise(resolve => setTimeout(resolve, attempts === 0 ? 500 : 1000));
          
          // Vérifier que l'utilisateur existe
          const { data: { user: currentUser }, error: userError } = await supabase.auth.getUser();
          
          if (!userError && currentUser) {
            user = currentUser;
            console.log('[ResetPasswordHandler] Token valide, utilisateur:', user.email);
            break;
          }

          attempts++;
        }

        // Si on n'a pas trouvé d'utilisateur après plusieurs tentatives, 
        // on affiche quand même le formulaire et on laisse le formulaire gérer la validation
        // (le formulaire vérifiera la session au moment de la soumission)
        if (!user) {
          console.warn('[ResetPasswordHandler] Session non disponible immédiatement, affichage du formulaire quand même');
          // On affiche le formulaire et on laisse le formulaire gérer la validation
          setIsValidToken(true);
        } else {
          setIsValidToken(true);
        }
      } catch (err) {
        console.error('[ResetPasswordHandler] Erreur inattendue:', err);
        // En cas d'erreur, on affiche quand même le formulaire
        // Le formulaire gérera la validation au moment de la soumission
        setIsValidToken(true);
      } finally {
        setIsValidating(false);
      }
    }

    validateToken();
  }, [router, searchParams]);

  if (isValidating) {
    return (
      <div className="space-y-4">
        <div className="rounded-md bg-zinc-50 p-3 text-sm text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
          Vérification du lien de réinitialisation...
        </div>
      </div>
    );
  }

  if (!isValidToken || error) {
    return (
      <div className="space-y-4">
        <div className="rounded-md bg-rose-50 p-3 text-sm text-rose-700 dark:bg-rose-900/20 dark:text-rose-400">
          {error || 'Lien de réinitialisation invalide ou expiré. Veuillez demander un nouveau lien.'}
        </div>
        <a
          href="/login/forgot-password"
          className="block w-full rounded-md bg-black py-2 text-center text-white transition hover:bg-zinc-800"
        >
          Demander un nouveau lien
        </a>
      </div>
    );
  }

  return <ResetPasswordForm />;
}

