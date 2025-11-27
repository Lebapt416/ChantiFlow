import { redirect } from 'next/navigation';
import Link from 'next/link';
import { LayoutDashboard, User } from 'lucide-react';
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

  // Vérifier que l'utilisateur est connecté
  if (!user) {
    redirect('/login?redirect=/analytics/profile');
  }

  // Vérifier que l'utilisateur est le compte autorisé (par ID ou email)
  const authorizedUserId = 'e78e437e-a817-4da2-a091-a7f4e5e02583';
  if (user.id !== authorizedUserId && user.email !== 'bcb83@icloud.com') {
    redirect('/login?error=unauthorized');
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950 text-white">
      {/* Menu latéral */}
      <aside className="fixed inset-y-0 left-0 z-20 w-16 flex-col items-center border-r border-zinc-800 bg-black/80 px-0 py-8 shadow-xl backdrop-blur flex">
        <nav className="flex flex-1 flex-col items-center gap-2 w-full">
          <Link
            href="/analytics"
            className="group/item relative flex items-center justify-center w-14 h-14 rounded-xl transition-all duration-200 text-white hover:text-white"
            title="Analytics"
          >
            <span className="absolute rounded-xl transition-all duration-200 top-0 bottom-0 left-2 right-0 bg-black/50 group-hover/item:bg-black/70 group-hover/item:left-3"></span>
            <span className="relative z-10">
              <LayoutDashboard size={26} strokeWidth={3} className="group-hover/item:scale-110 transition-transform duration-200" />
            </span>
          </Link>
          <Link
            href="/analytics/profile"
            className="group/item relative flex items-center justify-center w-14 h-14 rounded-xl transition-all duration-200 bg-white text-black shadow-lg shadow-white/20"
            title="Profil"
          >
            <span className="absolute rounded-xl transition-all duration-200 inset-0 bg-white"></span>
            <span className="relative z-10">
              <User size={26} strokeWidth={3.5} />
            </span>
          </Link>
        </nav>
      </aside>
      <div className="ml-16">
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
                    C&apos;est moi le patron babyyy
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
