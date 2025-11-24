'use client';

import { useEffect, useState } from 'react';
import { checkAIStatus } from '@/lib/ai/status';

type ClientStatus =
  | { status: 'loading'; message: string }
  | { status: 'missing_config'; message: string }
  | { status: 'connected'; message: string; url?: string }
  | { status: 'error'; message: string };

const STATUS_STYLES: Record<string, string> = {
  loading: 'bg-zinc-400',
  connected: 'bg-emerald-500',
  error: 'bg-rose-500',
  missing_config: 'bg-amber-400',
};

function maskUrl(url?: string) {
  if (!url) {
    return '';
  }
  try {
    const { origin } = new URL(url);
    if (origin.length <= 12) {
      return origin;
    }
    return `${origin.slice(0, 8)}…${origin.slice(-4)}`;
  } catch {
    return url.length > 12 ? `${url.slice(0, 8)}…${url.slice(-4)}` : url;
  }
}

export function AIStatusBadge() {
  const [status, setStatus] = useState<ClientStatus>({
    status: 'loading',
    message: 'Vérification en cours…',
  });

  useEffect(() => {
    let active = true;
    checkAIStatus()
      .then((result) => {
        if (!active) return;
        setStatus(result);
      })
      .catch(() => {
        if (!active) return;
        setStatus({ status: 'error', message: 'Erreur de connexion' });
      });

    return () => {
      active = false;
    };
  }, []);

  const indicatorClass = STATUS_STYLES[status.status] ?? STATUS_STYLES.loading;

  const tooltip =
    status.status === 'connected' ? `API: ${maskUrl(status.url)}` : status.message;

  const labelMap: Record<string, string> = {
    loading: 'IA: vérification…',
    connected: 'IA connectée',
    error: 'IA hors ligne',
    missing_config: 'IA non configurée',
  };

  return (
    <div
      className="inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-white px-3 py-1 text-xs font-medium text-zinc-600 shadow-sm dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300"
      title={tooltip}
    >
      <span className={`h-2.5 w-2.5 rounded-full ${indicatorClass}`} aria-hidden />
      <span>{labelMap[status.status] ?? status.message}</span>
    </div>
  );
}

