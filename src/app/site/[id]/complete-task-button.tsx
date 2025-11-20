'use client';

import { useFormStatus } from 'react-dom';
import { completeTaskAction } from './actions';

function CompleteTaskSubmit({ disabled }: { disabled?: boolean }) {
  const { pending } = useFormStatus();
  return (
    <button
      type="submit"
      className="rounded-full border border-emerald-600 px-3 py-1 text-xs font-semibold text-emerald-700 transition hover:bg-emerald-50 disabled:opacity-60 dark:border-emerald-400 dark:text-emerald-200 dark:hover:bg-emerald-900/30"
      disabled={pending || disabled}
    >
      {pending ? '...' : 'Terminer'}
    </button>
  );
}

type Props = {
  siteId: string;
  taskId: string;
  disabled?: boolean;
};

export function CompleteTaskButton({ siteId, taskId, disabled }: Props) {
  return (
    <form action={completeTaskAction}>
      <input type="hidden" name="siteId" value={siteId} />
      <input type="hidden" name="taskId" value={taskId} />
      <CompleteTaskSubmit disabled={disabled} />
    </form>
  );
}

