import { createBrowserClient } from '@supabase/ssr';

export function createSupabaseBrowserClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!url || !anonKey) {
    const missing = [];
    if (!url) missing.push('NEXT_PUBLIC_SUPABASE_URL');
    if (!anonKey) missing.push('NEXT_PUBLIC_SUPABASE_ANON_KEY');
    
    throw new Error(
      `Supabase est mal configuré côté client. Variables manquantes : ${missing.join(', ')}. ` +
      `Veuillez ajouter ces variables dans votre fichier .env.local. ` +
      `Voir ENV_SETUP.md pour plus d'informations.`
    );
  }

  // createBrowserClient gère automatiquement la persistance via localStorage
  // Ce qui est parfait pour les PWA
  return createBrowserClient(url, anonKey);
}

