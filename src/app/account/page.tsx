import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { signOutAction } from '../actions';
import { SignOutButton } from './sign-out-button';

export const metadata = {
  title: 'Mon compte | ChantiFlow',
};

type Plan = 'basic' | 'plus' | 'pro';

async function getUserPlan(userId: string): Promise<Plan> {
  // Pour l'instant, on détermine le plan selon le nombre de chantiers
  // Plus tard, cela viendra de la base de données ou de Stripe
  const supabase = await createSupabaseServerClient();
  const { data: sites } = await supabase
    .from('sites')
    .select('id')
    .eq('created_by', userId);

  const siteCount = sites?.length ?? 0;

  if (siteCount === 0) {
    return 'basic';
  } else if (siteCount <= 1) {
    return 'basic';
  } else if (siteCount <= 5) {
    return 'plus';
  } else {
    return 'pro';
  }
}

export default async function AccountPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const plan = await getUserPlan(user.id);

  const planNames: Record<Plan, string> = {
    basic: 'Basic',
    plus: 'Plus',
    pro: 'Pro',
  };

  const planFeatures: Record<Plan, string[]> = {
    basic: ['1 chantier actif', 'Planification IA basique', 'QR codes', 'Support par email'],
    plus: ['Jusqu\'à 5 chantiers actifs', 'Planification IA avancée', 'Analytics', 'Support prioritaire'],
    pro: ['Chantiers illimités', 'Multi-utilisateurs', 'API personnalisée', 'Support 24/7'],
  };

  const planPrices: Record<Plan, string> = {
    basic: 'Gratuit',
    plus: '29€/mois',
    pro: '79€/mois',
  };

  return (
    <AppShell
      heading="Mon compte"
      subheading="Gérez vos informations et votre abonnement"
      userEmail={user.email}
    >
      <div className="space-y-6">
        {/* Informations utilisateur */}
        <section className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
            Informations de connexion
          </h2>
          <div className="mt-4 space-y-3">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
                Email
              </p>
              <p className="mt-1 text-sm font-medium text-zinc-900 dark:text-white">
                {user.email}
              </p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
                ID utilisateur
              </p>
              <p className="mt-1 text-xs font-mono text-zinc-500 dark:text-zinc-400">
                {user.id}
              </p>
            </div>
          </div>
        </section>

        {/* Plan actuel */}
        <section className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">Plan actuel</h2>
          <div className="mt-4">
            <div className="flex items-center justify-between rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800">
              <div>
                <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                  {planNames[plan]}
                </p>
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  {planPrices[plan]}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-zinc-500 dark:text-zinc-400">Fonctionnalités :</p>
                <ul className="mt-1 space-y-1 text-xs text-zinc-600 dark:text-zinc-300">
                  {planFeatures[plan].slice(0, 2).map((feature, index) => (
                    <li key={index}>• {feature}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Changer de plan */}
        <section className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">Changer de plan</h2>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            Mettez à niveau votre abonnement pour accéder à plus de fonctionnalités.
          </p>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            {(['basic', 'plus', 'pro'] as Plan[]).map((p) => (
              <div
                key={p}
                className={`rounded-xl border p-4 ${
                  p === plan
                    ? 'border-emerald-500 bg-emerald-50 dark:border-emerald-400 dark:bg-emerald-900/20'
                    : 'border-zinc-200 dark:border-zinc-700'
                }`}
              >
                <div className="mb-3">
                  <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                    {planNames[p]}
                  </p>
                  <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                    {planPrices[p]}
                  </p>
                </div>
                <ul className="mb-4 space-y-1 text-xs text-zinc-600 dark:text-zinc-300">
                  {planFeatures[p].slice(0, 3).map((feature, index) => (
                    <li key={index}>• {feature}</li>
                  ))}
                </ul>
                <button
                  type="button"
                  disabled
                  className={`w-full rounded-lg px-3 py-2 text-xs font-semibold transition ${
                    p === plan
                      ? 'bg-emerald-600 text-white dark:bg-emerald-400 dark:text-zinc-900'
                      : p === 'basic'
                        ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-900'
                        : 'bg-zinc-200 text-zinc-900 dark:bg-zinc-700 dark:text-white'
                  } disabled:cursor-not-allowed disabled:opacity-50`}
                >
                  {p === plan ? 'Plan actuel' : 'Choisir'}{' '}
                  {p !== plan && '(Stripe bientôt disponible)'}
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* Déconnexion */}
        <section className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">Déconnexion</h2>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            Déconnectez-vous de votre compte ChantiFlow.
          </p>
          <div className="mt-4">
            <SignOutButton />
          </div>
        </section>
      </div>
    </AppShell>
  );
}

