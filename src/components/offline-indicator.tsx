'use client';

import { WifiOff, Wifi, Clock, CheckCircle2 } from 'lucide-react';
import { useOfflineSync } from '@/hooks/useOfflineSync';

export function OfflineIndicator() {
  const { isOnline, pendingCount, isSyncing } = useOfflineSync();

  if (isOnline && pendingCount === 0) {
    return null;
  }

  return (
    <div
      className={`fixed bottom-4 right-4 z-50 rounded-lg border px-4 py-3 shadow-lg transition-all ${
        isOnline
          ? 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950/50'
          : 'border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/50'
      }`}
    >
      <div className="flex items-center gap-3">
        {isOnline ? (
          <>
            {isSyncing ? (
              <>
                <Clock className="h-5 w-5 animate-spin text-blue-600 dark:text-blue-400" />
                <div>
                  <p className="text-sm font-semibold text-blue-900 dark:text-blue-100">
                    Synchronisation en cours...
                  </p>
                  <p className="text-xs text-blue-700 dark:text-blue-300">
                    {pendingCount} rapport{pendingCount > 1 ? 's' : ''} en attente
                  </p>
                </div>
              </>
            ) : pendingCount > 0 ? (
              <>
                <CheckCircle2 className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                <div>
                  <p className="text-sm font-semibold text-blue-900 dark:text-blue-100">
                    En ligne
                  </p>
                  <p className="text-xs text-blue-700 dark:text-blue-300">
                    {pendingCount} rapport{pendingCount > 1 ? 's' : ''} synchronisé{pendingCount > 1 ? 's' : ''}
                  </p>
                </div>
              </>
            ) : null}
          </>
        ) : (
          <>
            <WifiOff className="h-5 w-5 text-amber-600 dark:text-amber-400" />
            <div>
              <p className="text-sm font-semibold text-amber-900 dark:text-amber-100">
                Mode Hors-ligne
              </p>
              <p className="text-xs text-amber-700 dark:text-amber-300">
                {pendingCount > 0
                  ? `${pendingCount} rapport${pendingCount > 1 ? 's' : ''} en attente de synchronisation`
                  : 'Vos rapports seront sauvegardés localement'}
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

