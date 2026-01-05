'use client';

import { WifiOff, Wifi, Clock, CheckCircle2, AlertCircle } from 'lucide-react';
import { useOfflineSync } from '@/hooks/useOfflineSync';
import { useState, useEffect } from 'react';

export function OfflineIndicator() {
  const { isOnline, pendingCount, isSyncing, getPendingReports } = useOfflineSync();
  const [priorityCounts, setPriorityCounts] = useState<{ high: number; medium: number; low: number }>({
    high: 0,
    medium: 0,
    low: 0,
  });

  // Mettre à jour les compteurs par priorité (optimisé: moins fréquent)
  useEffect(() => {
    // Ne pas exécuter immédiatement, attendre que le composant soit monté
    if (typeof window === 'undefined') return;
    
    // Si pas de rapports en attente, ne rien faire
    if (pendingCount === 0) {
      setPriorityCounts({ high: 0, medium: 0, low: 0 });
      return;
    }
    
    let isMounted = true;
    
    const updatePriorityCounts = async () => {
      if (!isMounted || pendingCount === 0) return;
      
      try {
        // Calculer directement depuis IndexedDB pour éviter les dépendances
        const db = await new Promise<IDBDatabase>((resolve, reject) => {
          const request = indexedDB.open('ChantiFlowOffline', 2);
          request.onsuccess = () => resolve(request.result);
          request.onerror = () => reject(request.error);
        });
        
        const transaction = db.transaction(['pendingReports'], 'readonly');
        const store = transaction.objectStore('pendingReports');
        const request = store.getAll();
        
        request.onsuccess = () => {
          if (!isMounted) return;
          const reports = request.result as Array<{ priority: string }>;
          const counts = {
            high: reports.filter((r) => r.priority === 'high').length,
            medium: reports.filter((r) => r.priority === 'medium').length,
            low: reports.filter((r) => r.priority === 'low').length,
          };
          setPriorityCounts(counts);
        };
      } catch (error) {
        // Ignorer silencieusement les erreurs
      }
    };

    // Délai initial pour ne pas bloquer le rendu
    const timeoutId = setTimeout(() => {
      updatePriorityCounts();
    }, 3000); // Augmenté à 3s pour ne pas bloquer le FCP
    
    // Intervalle très long (30s) pour réduire la charge
    const interval = setInterval(updatePriorityCounts, 30000);
    
    return () => {
      isMounted = false;
      clearTimeout(timeoutId);
      clearInterval(interval);
    };
  }, [pendingCount]); // Seulement pendingCount comme dépendance

  if (isOnline && pendingCount === 0) {
    return null;
  }

  const hasHighPriority = priorityCounts.high > 0;

  return (
    <div
      className={`fixed bottom-4 right-4 z-50 rounded-lg border px-4 py-3 shadow-lg transition-all ${
        isOnline
          ? hasHighPriority
            ? 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950/50'
            : 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950/50'
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
                    {hasHighPriority && ` (${priorityCounts.high} prioritaire${priorityCounts.high > 1 ? 's' : ''})`}
                  </p>
                </div>
              </>
            ) : pendingCount > 0 ? (
              <>
                {hasHighPriority ? (
                  <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
                ) : (
                  <CheckCircle2 className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                )}
                <div>
                  <p className="text-sm font-semibold text-blue-900 dark:text-blue-100">
                    En ligne
                  </p>
                  <p className="text-xs text-blue-700 dark:text-blue-300">
                    {pendingCount} rapport{pendingCount > 1 ? 's' : ''} en attente de synchronisation
                    {hasHighPriority && (
                      <span className="ml-1 font-semibold text-red-600 dark:text-red-400">
                        ({priorityCounts.high} prioritaire{priorityCounts.high > 1 ? 's' : ''})
                      </span>
                    )}
                  </p>
                  {priorityCounts.medium > 0 || priorityCounts.low > 0 ? (
                    <p className="mt-1 text-xs text-blue-600 dark:text-blue-400">
                      {priorityCounts.medium > 0 && `${priorityCounts.medium} moyen${priorityCounts.medium > 1 ? 's' : ''}`}
                      {priorityCounts.medium > 0 && priorityCounts.low > 0 && ' • '}
                      {priorityCounts.low > 0 && `${priorityCounts.low} faible${priorityCounts.low > 1 ? 's' : ''}`}
                    </p>
                  ) : null}
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
                {pendingCount > 0 ? (
                  <>
                    <span className="font-semibold">{pendingCount} rapport{pendingCount > 1 ? 's' : ''}</span> en attente de synchronisation
                    {hasHighPriority && (
                      <span className="ml-1 font-semibold text-red-600 dark:text-red-400">
                        ({priorityCounts.high} prioritaire{priorityCounts.high > 1 ? 's' : ''})
                      </span>
                    )}
                  </>
                ) : (
                  'Vos rapports seront sauvegardés localement'
                )}
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

