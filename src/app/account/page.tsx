import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { SignOutButton } from './sign-out-button';
import { ChangePlanButton } from './change-plan-button';
import { getUserPlan, getUserAddOns, getPlanLimits, type Plan } from '@/lib/plans';
import { AddOnsSection } from './add-ons-section';
import { isAdmin } from '@/lib/admin';
import { SystemTestButton } from './system-test-button';

export const metadata = {
  title: 'Mon compte | ChantiFlow',
};

export default async function AccountPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  const plan = await getUserPlan(user);
  const addOns = getUserAddOns(user);
  const limits = getPlanLimits(plan, addOns);
  
  // Vérification admin
  const userIsAdmin = isAdmin(user.email);

  const planNames: Record<Plan, string> = {
    basic: 'Basic',
    plus: 'Plus',
    pro: 'Pro',
  };

  const planFeatures: Record<Plan, string[]> = {
    basic: ['1 chantier actif', '3 employés max', 'Planification IA basique', 'QR codes', 'Support par email'],
    plus: ['Jusqu\'à 5 chantiers actifs', '7 employés max', 'Planification IA avancée', 'Analytics', 'Support prioritaire'],
    pro: ['Chantiers illimités', 'Employés illimités', 'Multi-utilisateurs', 'API personnalisée', 'Support 24/7'],
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
                <p className="text-xs text-zinc-500 dark:text-zinc-400">Limites actuelles :</p>
                <ul className="mt-1 space-y-1 text-xs text-zinc-600 dark:text-zinc-300">
                  <li>
                    • {limits.maxSites === Infinity ? 'Chantiers illimités' : `${limits.maxSites} chantier${limits.maxSites > 1 ? 's' : ''} max`}
                    {addOns.extra_sites ? ` (+${addOns.extra_sites * 2} via add-ons)` : ''}
                  </li>
                  <li>
                    • {limits.maxWorkers === Infinity ? 'Employés illimités' : `${limits.maxWorkers} employé${limits.maxWorkers > 1 ? 's' : ''} max`}
                    {addOns.extra_workers ? ` (+${addOns.extra_workers * 5} via add-ons)` : ''}
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Add-ons */}
        <AddOnsSection user={user} currentAddOns={addOns} plan={plan} />

        {/* Changer de plan */}
        <section className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">Changer de plan</h2>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            Mettez à niveau votre abonnement pour accéder à plus de fonctionnalités.
          </p>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            {(['basic', 'plus', 'pro'] as Plan[]).map((p) => {
              const planLimits = getPlanLimits(p);
              return (
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
                    <li>• {planLimits.maxSites === Infinity ? 'Chantiers illimités' : `${planLimits.maxSites} chantier${planLimits.maxSites > 1 ? 's' : ''} max`}</li>
                    <li>• {planLimits.maxWorkers === Infinity ? 'Employés illimités' : `${planLimits.maxWorkers} employé${planLimits.maxWorkers > 1 ? 's' : ''} max`}</li>
                    <li>• {planFeatures[p][2]}</li>
                  </ul>
                  <ChangePlanButton plan={p} currentPlan={plan} userEmail={user.email} />
                </div>
              );
            })}
          </div>
        </section>

        {/* Tests système (admin uniquement) */}
        <section className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">Tests système</h2>
          {userIsAdmin ? (
            <>
              <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                Vérifiez le bon fonctionnement de l'application et obtenez un rapport détaillé des erreurs.
              </p>
              <div className="mt-4">
                <SystemTestButton />
              </div>
            </>
          ) : (
            <>
              <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                Cette fonctionnalité est réservée aux administrateurs.
              </p>
              <div className="mt-4 rounded-lg border border-yellow-200 bg-yellow-50 p-3 dark:border-yellow-800 dark:bg-yellow-900/20">
                <p className="text-xs text-yellow-800 dark:text-yellow-200">
                  <strong>Email actuel :</strong> {user.email}
                </p>
                <p className="mt-1 text-xs text-yellow-800 dark:text-yellow-200">
                  <strong>Pour activer :</strong> Ajoutez votre email dans la variable d'environnement <code className="rounded bg-yellow-100 px-1 dark:bg-yellow-900/40">ADMIN_EMAILS</code>
                </p>
                <p className="mt-1 text-xs text-yellow-800 dark:text-yellow-200">
                  Format : <code className="rounded bg-yellow-100 px-1 dark:bg-yellow-900/40">ADMIN_EMAILS={user.email}</code>
                </p>
              </div>
            </>
          )}
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

