import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { AIPlanningSection } from '@/components/ai-planning-section';

export const metadata = {
  title: 'IA de Planification | ChantiFlow',
};

export default async function AIPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const { data: sites } = await supabase
    .from('sites')
    .select('id, name, deadline')
    .eq('created_by', user.id)
    .order('created_at', { ascending: false });

  return (
    <AppShell
      heading="IA de Planification"
      subheading="Génère un planning optimisé avec l'intelligence artificielle"
      userEmail={user.email}
      primarySite={sites?.[0] ?? null}
    >
      <AIPlanningSection sites={sites ?? []} />
    </AppShell>
  );
}

