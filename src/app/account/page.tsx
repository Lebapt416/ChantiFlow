import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { AppShell } from '@/components/app-shell';
import { SignOutButton } from './sign-out-button';
import { ChangePlanButton } from './change-plan-button';
import { getUserPlan, getUserAddOns, getPlanLimits, type Plan } from '@/lib/plans';
import { AddOnsSection } from './add-ons-section';

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
        <section className="rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
          <h2 className="font-serif text-[22px] text-ink dark:text-paper">
            Informations de connexion
          </h2>
          <div className="mt-4 space-y-3">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-ink-3">
                Email
              </p>
              <p className="mt-1 text-sm font-medium text-ink dark:text-paper">
                {user.email}
              </p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-ink-3">
                ID utilisateur
              </p>
              <p className="mt-1 text-xs font-mono text-ink-3">
                {user.id}
              </p>
            </div>
          </div>
        </section>

        {/* Plan actuel */}
        <section className="rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
          <h2 className="font-serif text-[22px] text-ink dark:text-paper">Plan actuel</h2>
          <div className="mt-4">
            <div className="flex items-center justify-between border border-rule-soft bg-paper-2 p-4 dark:border-rule">
              <div>
                <p className="text-sm font-semibold text-ink dark:text-paper">
                  {planNames[plan]}
                </p>
                <p className="mt-1 text-xs text-ink-3">
                  {planPrices[plan]}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-ink-3">Limites actuelles :</p>
                <ul className="mt-1 space-y-1 text-xs text-ink-2">
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
        <section className="rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
          <h2 className="font-serif text-[22px] text-ink dark:text-paper">Changer de plan</h2>
          <p className="mt-1 text-sm text-ink-3">
            Mettez à niveau votre abonnement pour accéder à plus de fonctionnalités.
          </p>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            {(['basic', 'plus', 'pro'] as Plan[]).map((p) => {
              const planLimits = getPlanLimits(p);
              return (
                <div
                  key={p}
                  className={`rounded border p-4 ${
                    p === plan
                      ? 'border-orange bg-paper-2 dark:border-orange dark:bg-paper-2'
                      : 'border-rule-soft dark:border-rule'
                  }`}
                >
                  <div className="mb-3">
                    <p className="text-sm font-semibold text-ink dark:text-paper">
                      {planNames[p]}
                    </p>
                    <p className="mt-1 text-xs text-ink-3">
                      {planPrices[p]}
                    </p>
                  </div>
                  <ul className="mb-4 space-y-1 text-xs text-ink-2">
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

        {/* Déconnexion */}
        <section className="rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink">
          <h2 className="font-serif text-[22px] text-ink dark:text-paper">Déconnexion</h2>
          <p className="mt-1 text-sm text-ink-3">
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
