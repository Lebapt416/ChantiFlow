import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';

export const metadata = {
  title: 'Profil | Analytics ChantiFlow',
  description: 'Profil administrateur',
};

export const dynamic = 'force-dynamic';

export default async function AnalyticsProfilePage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  // Vérifier que l'utilisateur est connecté et qu'il s'agit du compte autorisé
  if (!user || user.email !== 'bcb83@icloud.com') {
    redirect('/login');
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950 text-white">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-2xl mx-auto">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-8 backdrop-blur">
            <h1 className="text-4xl font-bold mb-4">Profil Administrateur</h1>
            <div className="space-y-6">
              <div>
                <p className="text-sm text-zinc-400 mb-2">Email</p>
                <p className="text-lg font-semibold text-white">{user.email}</p>
              </div>
              <div>
                <p className="text-sm text-zinc-400 mb-2">Description</p>
                <p className="text-2xl font-bold text-emerald-400">
                  C'est moi le patron babyyy
                </p>
              </div>
              <div>
                <p className="text-sm text-zinc-400 mb-2">Rôle</p>
                <p className="text-lg font-semibold text-white">Administrateur Analytics</p>
              </div>
              <div>
                <p className="text-sm text-zinc-400 mb-2">Date de création du compte</p>
                <p className="text-lg text-white">
                  {new Date(user.created_at).toLocaleDateString('fr-FR', {
                    day: '2-digit',
                    month: 'long',
                    year: 'numeric',
                  })}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}

