import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export const dynamic = 'force-dynamic';

export default async function Home() {
  // Vérifier si l'utilisateur est déjà connecté
  try {
    const supabase = await createSupabaseServerClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (user) {
      // Si connecté, rediriger vers la page appropriée
      const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
      if (user.id === authorizedUserId || user.email === 'bcb83@icloud.com') {
        redirect('/analytics');
      } else {
        redirect('/home');
      }
    } else {
      // Si pas connecté, aller à la landing page
      redirect('/landing');
    }
  } catch {
    // En cas d'erreur, aller à la landing page
    redirect('/landing');
  }
}
