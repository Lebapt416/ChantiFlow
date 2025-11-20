import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export const dynamic = 'force-dynamic';

export default async function Home() {
  try {
    const supabase = await createSupabaseServerClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();

    redirect(session ? '/home' : '/login');
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Erreur inconnue';
    
    // Afficher une page d'erreur claire si les variables d'environnement manquent
    if (message.includes('Supabase est mal configuré') || message.includes('Service role')) {
      return (
        <div className="min-h-screen bg-zinc-50 flex items-center justify-center px-4 dark:bg-zinc-950">
          <div className="max-w-2xl w-full rounded-2xl border border-rose-200 bg-white p-8 text-center shadow-lg dark:border-rose-900/60 dark:bg-zinc-900">
            <h1 className="text-2xl font-semibold text-zinc-900 dark:text-white mb-4">
              ⚠️ Configuration manquante
            </h1>
            <p className="text-zinc-600 dark:text-zinc-400 mb-6">
              Les variables d'environnement Supabase ne sont pas configurées sur Vercel.
            </p>
            <div className="text-left bg-zinc-50 dark:bg-zinc-800 p-4 rounded-lg mb-6">
              <p className="text-sm font-semibold mb-2 text-zinc-900 dark:text-white">
                Pour corriger :
              </p>
              <ol className="text-sm text-zinc-600 dark:text-zinc-300 space-y-2 list-decimal list-inside">
                <li>Allez sur Vercel → Settings → Environment Variables</li>
                <li>Ajoutez NEXT_PUBLIC_SUPABASE_URL</li>
                <li>Ajoutez NEXT_PUBLIC_SUPABASE_ANON_KEY</li>
                <li>Ajoutez SUPABASE_SERVICE_ROLE_KEY</li>
                <li>Redéployez votre application</li>
              </ol>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Consultez VERCEL_SETUP.md pour plus de détails
            </p>
          </div>
        </div>
      );
    }
    
    // Autre erreur
    throw error;
  }
}
