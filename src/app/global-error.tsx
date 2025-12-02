'use client';

import { useEffect } from 'react';
import { logger } from '@/lib/logger';

interface GlobalErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

/**
 * Composant de gestion d'erreur globale pour Next.js 16 App Router
 * Attrape les erreurs non gérées au niveau racine de l'application
 * 
 * @see https://nextjs.org/docs/app/api-reference/file-conventions/error-handling#global-errorjs
 */
export default function GlobalError({ error, reset }: GlobalErrorProps) {
  useEffect(() => {
    // Logger l'erreur avec le logger structuré
    logger.error('Erreur globale non gérée', {
      errorName: error.name,
      errorMessage: error.message,
      digest: error.digest,
      stack: error.stack,
    }, error);

    // Envoyer à Sentry si disponible (côté client)
    if (typeof window !== 'undefined') {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const Sentry = (window as any).Sentry;
        if (Sentry) {
          Sentry.captureException(error, {
            contexts: {
              error: {
                digest: error.digest,
              },
            },
          });
        }
      } catch {
        // Sentry non disponible - ignorer silencieusement
      }
    }
  }, [error]);

  return (
    <html lang="fr">
      <body>
        <div className="flex min-h-screen flex-col items-center justify-center bg-zinc-50 px-4 dark:bg-zinc-900">
          <div className="w-full max-w-md space-y-6 rounded-2xl border border-zinc-200 bg-white p-8 shadow-lg dark:border-zinc-800 dark:bg-zinc-950">
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-rose-100 dark:bg-rose-900/30">
                <span className="text-2xl">⚠️</span>
              </div>
              <div>
                <h1 className="text-xl font-semibold text-zinc-900 dark:text-white">
                  Erreur inattendue
                </h1>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  Une erreur s&apos;est produite
                </p>
              </div>
            </div>

            <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 dark:border-rose-800 dark:bg-rose-900/20">
              <p className="text-sm font-medium text-rose-900 dark:text-rose-200">
                {error.message || "Une erreur inattendue s'est produite"}
              </p>
              {process.env.NODE_ENV === 'development' && error.stack && (
                <details className="mt-3">
                  <summary className="cursor-pointer text-xs text-rose-700 dark:text-rose-300">
                    Détails techniques
                  </summary>
                  <pre className="mt-2 max-h-40 overflow-auto rounded bg-rose-100 p-2 text-xs text-rose-900 dark:bg-rose-900/40 dark:text-rose-100">
                    {error.stack}
                  </pre>
                </details>
              )}
            </div>

            <div className="flex gap-3">
              <button
                onClick={reset}
                className="flex-1 rounded-lg bg-emerald-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 dark:bg-emerald-500 dark:hover:bg-emerald-600"
              >
                Réessayer
              </button>
              <button
                onClick={() => {
                  window.location.href = '/';
                }}
                className="flex-1 rounded-lg border border-zinc-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-50 focus:outline-none focus:ring-2 focus:ring-zinc-500 focus:ring-offset-2 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
              >
                Retour à l&apos;accueil
              </button>
            </div>

            {error.digest && (
              <p className="text-center text-xs text-zinc-400">
                Code d&apos;erreur: {error.digest}
              </p>
            )}
          </div>
        </div>
      </body>
    </html>
  );
}

