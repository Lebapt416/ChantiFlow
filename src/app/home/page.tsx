import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export const metadata = {
  title: 'Accueil | ChantiFlow',
};

export default async function HomePage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: sites } = await supabase
    .from('sites')
    .select('id, name, deadline, created_at')
    .eq('created_by', user.id)
    .order('created_at', { ascending: false });

  // Rediriger vers le premier chantier s'il existe, sinon vers le dashboard
  if (sites && sites.length > 0) {
    redirect(`/site/${sites[0].id}`);
  } else {
    redirect('/dashboard');
  }
}

