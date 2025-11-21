'use client';

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { Trash2 } from 'lucide-react';
import { deleteWorkerAction } from '@/app/team/actions';

type Props = {
  workerId: string;
  workerName: string;
};

export function DeleteWorkerButton({ workerId, workerName }: Props) {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const [showConfirm, setShowConfirm] = useState(false);

  function handleDelete() {
    if (!showConfirm) {
      setShowConfirm(true);
      return;
    }

    startTransition(async () => {
      const formData = new FormData();
      formData.append('workerId', workerId);
      const result = await deleteWorkerAction({}, formData);
      
      if (result.error) {
        alert(`Erreur: ${result.error}`);
        setShowConfirm(false);
      } else {
        // Rafraîchir la page pour afficher les changements
        router.refresh();
      }
    });
  }

  function handleCancel() {
    setShowConfirm(false);
  }

  if (showConfirm) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-xs text-rose-600 dark:text-rose-400">
          Confirmer la suppression de {workerName} ?
        </span>
        <button
          type="button"
          onClick={handleDelete}
          disabled={isPending}
          className="rounded-md bg-rose-600 px-2 py-1 text-xs font-medium text-white transition hover:bg-rose-700 disabled:cursor-not-allowed disabled:opacity-70"
        >
          {isPending ? 'Suppression...' : 'Oui'}
        </button>
        <button
          type="button"
          onClick={handleCancel}
          disabled={isPending}
          className="rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs font-medium text-zinc-700 transition hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-70 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
        >
          Non
        </button>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={handleDelete}
      disabled={isPending}
      className="rounded-md p-1.5 text-zinc-400 transition hover:bg-zinc-100 hover:text-rose-600 disabled:cursor-not-allowed disabled:opacity-70 dark:hover:bg-zinc-800 dark:hover:text-rose-400"
      title="Supprimer"
    >
      <Trash2 className="h-4 w-4" />
    </button>
  );
}

