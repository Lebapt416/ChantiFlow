'use client';

import { useState } from 'react';

type Props = {
  value: string;
};

export function CopyButton({ value }: Props) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={`rounded-full border px-3 py-1 text-xs font-semibold transition ${
        copied
          ? 'border-emerald-500 text-emerald-600 dark:border-emerald-400 dark:text-emerald-300'
          : 'border-zinc-200 text-zinc-700 hover:border-zinc-900 hover:text-zinc-900 dark:border-zinc-600 dark:text-zinc-200 dark:hover:border-white dark:hover:text-white'
      }`}
    >
      {copied ? '✓ Copié' : '📋'}
    </button>
  );
}

