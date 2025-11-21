'use client';

import { useState, useEffect, useTransition } from 'react';
import { useFormState } from 'react-dom';
import { submitReportAction } from '@/app/qr/[siteId]/actions';
import { Camera, Upload } from 'lucide-react';

type Props = {
  siteId: string;
  workerId: string;
};

export function ReportsTab({ siteId, workerId }: Props) {
  const [state, formAction] = useFormState(submitReportAction, {});
  const [isPending, startTransition] = useTransition();
  const [taskId, setTaskId] = useState('');
  const [description, setDescription] = useState('');
  const [markDone, setMarkDone] = useState(false);
  const [tasks, setTasks] = useState<Array<{ id: string; title: string }>>([]);
  const [workerEmail, setWorkerEmail] = useState('');

  // Charger les tâches et l'email du worker
  useEffect(() => {
    async function loadData() {
      const { createSupabaseBrowserClient } = await import('@/lib/supabase/client');
      const supabase = createSupabaseBrowserClient();

      // Charger les tâches
      const { data: tasksData } = await supabase
        .from('tasks')
        .select('id, title')
        .eq('site_id', siteId)
        .order('created_at', { ascending: true });

      if (tasksData) {
        setTasks(tasksData);
      }

      // Charger l'email du worker
      const { data: worker } = await supabase
        .from('workers')
        .select('email')
        .eq('id', workerId)
        .single();

      if (worker?.email) {
        setWorkerEmail(worker.email);
      }
    }

    loadData();
  }, [siteId, workerId]);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    formData.append('siteId', siteId);
    formData.append('email', workerEmail);

    startTransition(async () => {
      await formAction(formData);
      if (state?.success) {
        setDescription('');
        setMarkDone(false);
        setTaskId('');
      }
    });
  }

  return (
    <div className="p-4">
      <h2 className="text-lg font-semibold text-zinc-900 dark:text-white mb-4">
        Envoyer un rapport
      </h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="taskId" className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
            Tâche
          </label>
          <select
            id="taskId"
            name="taskId"
            value={taskId}
            onChange={(e) => setTaskId(e.target.value)}
            className="w-full rounded-lg border border-zinc-300 bg-white px-4 py-2 text-zinc-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-800 dark:text-white"
            required
          >
            <option value="">Sélectionner une tâche</option>
            {tasks.map((task) => (
              <option key={task.id} value={task.id}>
                {task.title}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="description" className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
            Description
          </label>
          <textarea
            id="description"
            name="description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={4}
            className="w-full rounded-lg border border-zinc-300 bg-white px-4 py-2 text-zinc-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-800 dark:text-white"
            placeholder="Décrivez l'avancement, les incidents, le matériel utilisé..."
          />
        </div>

        <div>
          <label htmlFor="photo" className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
            Photo (optionnel)
          </label>
          <div className="relative">
            <input
              type="file"
              id="photo"
              name="photo"
              accept="image/*"
              className="w-full rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm text-zinc-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-800 dark:text-white"
            />
            <Camera className="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-zinc-400" />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="mark_done"
            name="mark_done"
            checked={markDone}
            onChange={(e) => setMarkDone(e.target.checked)}
            className="h-4 w-4 rounded border-zinc-300 text-emerald-600 focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600"
          />
          <label htmlFor="mark_done" className="text-sm text-zinc-700 dark:text-zinc-300">
            Marquer la tâche comme terminée
          </label>
        </div>

        {state?.error && (
          <div className="rounded-lg bg-rose-50 p-3 text-sm text-rose-800 dark:bg-rose-900/30 dark:text-rose-300">
            {state.error}
          </div>
        )}

        {state?.success && (
          <div className="rounded-lg bg-emerald-50 p-3 text-sm text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300">
            ✓ Rapport envoyé avec succès
          </div>
        )}

        <button
          type="submit"
          disabled={isPending || !taskId}
          className="w-full rounded-lg bg-emerald-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-70 flex items-center justify-center gap-2"
        >
          <Upload className="h-5 w-5" />
          {isPending ? 'Envoi...' : 'Envoyer le rapport'}
        </button>
      </form>
    </div>
  );
}

