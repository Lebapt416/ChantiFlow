import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export default async function SystemTestLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser();

  // Vérifier que l'utilisateur est connecté
  if (authError || !user) {
    redirect('/login?redirect=/analytics/system-test');
  }

  // Vérifier que l'utilisateur est admin
  const { isAdmin } = await import('@/lib/admin');
  if (!isAdmin(user.email)) {
    redirect('/login?error=unauthorized');
  }

  return <>{children}</>;
}

