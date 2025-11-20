import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';

type Options = {
  allowCookieSetter?: boolean;
};

export async function createSupabaseServerClient(options?: Options) {
  const cookieStore = await cookies();
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  const allowCookieSetter = options?.allowCookieSetter ?? false;

  if (!url || !anonKey) {
    throw new Error(
      'Supabase est mal configuré. Renseigne NEXT_PUBLIC_SUPABASE_URL et NEXT_PUBLIC_SUPABASE_ANON_KEY.',
    );
  }

  return createServerClient(url, anonKey, {
    cookies: {
      get(name) {
        return cookieStore.get(name)?.value;
      },
      set(name, value, opts) {
        if (!allowCookieSetter) {
          return;
        }
        try {
          cookieStore.set({ name, value, ...opts });
        } catch {
          // ignore when Next bloque la mutation hors Server Action
        }
      },
      remove(name, opts) {
        if (!allowCookieSetter) {
          return;
        }
        try {
          cookieStore.set({ name, value: '', ...opts, maxAge: 0 });
        } catch {
          // ignore
        }
      },
    },
  });
}

