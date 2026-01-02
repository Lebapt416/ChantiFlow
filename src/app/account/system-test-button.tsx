'use client';

import Link from 'next/link';
import { Settings } from 'lucide-react';

export function SystemTestButton() {
  return (
    <Link
      href="/admin/system-test"
      className="flex items-center gap-2 rounded-lg border border-zinc-200 bg-white px-4 py-2 text-sm font-medium text-zinc-900 transition-colors hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-white dark:hover:bg-zinc-700"
    >
      <Settings className="h-4 w-4" />
      <span>Tests système</span>
    </Link>
  );
}

