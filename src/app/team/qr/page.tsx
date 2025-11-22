import { redirect } from 'next/navigation';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { TeamQrSection } from '../team-qr-section';

export const metadata = {
  title: 'QR Code d\'inscription | ChantiFlow',
};

export default async function TeamQrPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const appUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? 'http://localhost:3000';
  const qrUrl = `${appUrl.replace(/\/$/, '')}/team/join?userId=${user.id}`;

  return (
    <AppShell
      heading="QR code d'inscription à l'équipe"
      subheading="Partagez ce QR code pour permettre aux personnes de s'ajouter à votre équipe"
      userEmail={user.email}
      primarySite={null}
    >
      <div className="space-y-6">
        <TeamQrSection qrUrl={qrUrl} />
      </div>
    </AppShell>
  );
}

