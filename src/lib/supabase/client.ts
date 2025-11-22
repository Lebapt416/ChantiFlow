import { createBrowserClient } from '@supabase/ssr';

export function createSupabaseBrowserClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!url || !anonKey) {
    throw new Error('Supabase est mal configuré côté client.');
  }

  // createBrowserClient gère automatiquement la persistance via localStorage
  // Ce qui est parfait pour les PWA
  return createBrowserClient(url, anonKey);
}

