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

  // Vérifier que l'utilisateur est le compte autorisé (par ID ou email)
  const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
  if (user.id !== authorizedUserId && user.email !== 'bcb83@icloud.com') {
    redirect('/login?error=unauthorized');
  }

  return <>{children}</>;
}

