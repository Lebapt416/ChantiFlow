import { Suspense } from 'react';
import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { DashboardContent } from './dashboard-content';
import HomeLoading from './loading';

export const metadata = {
  title: 'Accueil | ChantiFlow',
};

export const revalidate = 60; // Cache ISR de 60 secondes

export default async function HomePage() {
  // UNIQUEMENT l'authentification au niveau principal (sécurité critique)
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user || !user.email) {
    redirect('/login');
  }

  // Afficher AppShell IMMÉDIATEMENT (non-bloquant)
  // Les données chargent ensuite via Suspense Streaming
  return (
    <AppShell
      heading="Mes chantiers"
      subheading="Gérez tous vos chantiers depuis cette page"
      userEmail={user.email}
    >
      <Suspense fallback={<HomeLoading />}>
        <DashboardContent 
          userId={user.id} 
          userEmail={user.email}
          userMetadata={user.user_metadata}
        />
      </Suspense>
    </AppShell>
  );
}

