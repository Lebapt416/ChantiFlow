'use client';

import { useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { CheckCircle2 } from 'lucide-react';
import { validateReportAction } from '../actions';

type Props = {
  reportId: string;
  taskId: string;
};

export function ValidateReportButton({ reportId, taskId }: Props) {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();

  const handleValidate = async () => {
    const formData = new FormData();
    formData.append('reportId', reportId);
    formData.append('taskId', taskId);

    startTransition(async () => {
      const result = await validateReportAction({}, formData);
      
      if (result.success) {
        // Recharger la page pour afficher le statut mis à jour
        router.refresh();
      } else if (result.error) {
        alert(result.error);
      }
    });
  };

  return (
    <button
      onClick={handleValidate}
      disabled={isPending}
      className="w-full rounded-lg bg-emerald-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-70 flex items-center justify-center gap-2"
    >
      <CheckCircle2 className="h-5 w-5" />
      {isPending ? 'Validation...' : 'Valider le rapport et terminer la tâche'}
    </button>
  );
}

