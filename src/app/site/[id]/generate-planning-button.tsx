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
        className="flex items-center gap-2 rounded-full bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-emerald-500 dark:hover:bg-emerald-600"
      >
        <Sparkles className="h-4 w-4" />
        {isPending ? 'Génération en cours...' : 'Générer le planning IA'}
      </button>

      {state.error && (
        <p className="text-sm text-rose-500 dark:text-rose-400">{state.error}</p>
      )}

      {state.success && state.planning && (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 dark:border-emerald-900/60 dark:bg-emerald-900/20">
          <p className="text-sm font-semibold text-emerald-900 dark:text-emerald-100">
            Planning généré avec succès !
          </p>
          <p className="mt-2 text-xs text-emerald-700 dark:text-emerald-300">
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

