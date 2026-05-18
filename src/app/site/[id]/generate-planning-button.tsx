'use client';

import { useState, useTransition } from 'react';
import { Sparkles } from 'lucide-react';
import { generatePlanningAction, type GeneratePlanningState } from './generate-planning-action';

type Props = {
  siteId: string;
};

export function GeneratePlanningButton({ siteId }: Props) {
  const [isPending, startTransition] = useTransition();
  const [state, setState] = useState<GeneratePlanningState>({});

  function handleClick() {
    startTransition(async () => {
      const result = await generatePlanningAction(siteId);
      setState(result);
    });
  }

  return (
    <div className="space-y-2">
      <button
        type="button"
        onClick={handleClick}
        disabled={isPending}
        className="flex items-center gap-2 rounded-full bg-orange px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-orange-dark disabled:cursor-not-allowed disabled:opacity-50 dark:bg-paper-20 dark:hover:bg-orange"
      >
        <Sparkles className="h-4 w-4" />
        {isPending ? 'Génération en cours...' : 'Générer le planning IA'}
      </button>

      {state.error && (
        <p className="text-sm text-rose-500 dark:text-rose-400">{state.error}</p>
      )}

      {state.success && state.planning && (
        <div className="rounded-xl border border-rule-soft bg-paper-2 p-4 dark:border-orange/60 dark:bg-paper-2">
          <p className="text-sm font-semibold text-ink dark:text-paper">
            Planning généré avec succès !
          </p>
          <p className="mt-2 text-xs text-ink dark:text-green">
            {state.planning.orderedTasks.length} tâches classées par ordre logique
          </p>
          {state.planning.warnings.length > 0 && (
            <div className="mt-3 space-y-1">
              {state.planning.warnings.map((warning, index) => (
                <p key={index} className="text-xs text-amber-700 dark:text-amber-300">
                  ⚠️ {warning}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

