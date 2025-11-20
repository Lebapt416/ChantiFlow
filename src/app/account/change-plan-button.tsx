'use client';

import { useActionState, useTransition } from 'react';
import { useFormStatus } from 'react-dom';
import { changePlanAction, type ChangePlanState } from './actions';

type Props = {
  plan: 'basic' | 'plus' | 'pro';
  currentPlan: 'basic' | 'plus' | 'pro';
};

function SubmitButton({ plan, currentPlan }: Props) {
  const { pending } = useFormStatus();
  const isCurrent = plan === currentPlan;

  return (
    <button
      type="submit"
      disabled={pending || isCurrent}
      className={`w-full rounded-lg px-3 py-2 text-xs font-semibold transition ${
        isCurrent
          ? 'bg-emerald-600 text-white dark:bg-emerald-400 dark:text-zinc-900'
          : plan === 'basic'
            ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-900'
            : 'bg-zinc-200 text-zinc-900 dark:bg-zinc-700 dark:text-white'
      } disabled:cursor-not-allowed disabled:opacity-50`}
    >
      {pending
        ? 'Changement...'
        : isCurrent
          ? 'Plan actuel'
          : 'Choisir ce plan'}
    </button>
  );
}

export function ChangePlanButton({ plan, currentPlan }: Props) {
  const [state, formAction] = useActionState(changePlanAction, {});

  return (
    <form action={formAction}>
      <input type="hidden" name="plan" value={plan} />
      <SubmitButton plan={plan} currentPlan={currentPlan} />
      {state?.error && (
        <p className="mt-1 text-xs text-rose-500">{state.error}</p>
      )}
      {state?.success && (
        <p className="mt-1 text-xs text-emerald-500">Plan mis à jour !</p>
      )}
    </form>
  );
}

