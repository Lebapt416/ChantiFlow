'use client';

import { useFormStatus } from 'react-dom';
import { completeTaskAction } from './actions';

function CompleteTaskSubmit({ disabled }: { disabled?: boolean }) {
  const { pending } = useFormStatus();
  return (
    <button
      type="submit"
      className="rounded-full border border-orange px-3 py-1 text-xs font-semibold text-ink transition hover:bg-paper-2 disabled:opacity-60 dark:border-orange dark:text-orange dark:hover:bg-paper-2"
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

