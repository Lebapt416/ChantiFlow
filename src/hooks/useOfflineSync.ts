'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

export type PendingReport = {
  id: string;
  siteId: string;
  taskId: string;
  email: string;
  name: string;
  role: string;
  description: string;
  photo?: File;
  markDone: boolean;
  createdAt: number;
  retryCount: number;
};

const DB_NAME = 'ChantiFlowOffline';
const DB_VERSION = 1;
const STORE_NAME = 'pendingReports';

/**
 * Hook pour gérer la synchronisation offline avec IndexedDB
 */
export function useOfflineSync() {
  const [isOnline, setIsOnline] = useState(true);
  const [pendingCount, setPendingCount] = useState(0);
  const [isSyncing, setIsSyncing] = useState(false);
  const dbRef = useRef<IDBDatabase | null>(null);

  // Initialiser IndexedDB
  const initDB = useCallback((): Promise<IDBDatabase> => {
    return new Promise((resolve, reject) => {
      if (dbRef.current) {
        resolve(dbRef.current);
        return;
      }

      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        dbRef.current = request.result;
        resolve(request.result);
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const objectStore = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
          objectStore.createIndex('createdAt', 'createdAt', { unique: false });
          objectStore.createIndex('siteId', 'siteId', { unique: false });
        }
      };
    });
  }, []);

  // Sauvegarder un rapport en attente
  const savePendingReport = useCallback(
    async (report: Omit<PendingReport, 'id' | 'createdAt' | 'retryCount'>): Promise<string> => {
      const db = await initDB();
      const id = crypto.randomUUID();
      const pendingReport: PendingReport = {
        ...report,
        id,
        createdAt: Date.now(),
        retryCount: 0,
      };

      return new Promise((resolve, reject) => {
        const transaction = db.transaction([STORE_NAME], 'readwrite');
        const store = transaction.objectStore(STORE_NAME);
        const request = store.add(pendingReport);

        request.onsuccess = () => {
          setPendingCount((prev) => prev + 1);
          resolve(id);
        };
        request.onerror = () => reject(request.error);
      });
    },
    [initDB]
  );

  // Récupérer tous les rapports en attente
  const getPendingReports = useCallback(async (): Promise<PendingReport[]> => {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onsuccess = () => {
        const reports = request.result as PendingReport[];
        resolve(reports.sort((a, b) => a.createdAt - b.createdAt));
      };
      request.onerror = () => reject(request.error);
    });
  }, [initDB]);

  // Supprimer un rapport en attente
  const removePendingReport = useCallback(
    async (id: string): Promise<void> => {
      const db = await initDB();
      return new Promise((resolve, reject) => {
        const transaction = db.transaction([STORE_NAME], 'readwrite');
        const store = transaction.objectStore(STORE_NAME);
        const request = store.delete(id);

        request.onsuccess = () => {
          setPendingCount((prev) => Math.max(0, prev - 1));
          resolve();
        };
        request.onerror = () => reject(request.error);
      });
    },
    [initDB]
  );

  // Mettre à jour le compteur de retry
  const incrementRetryCount = useCallback(
    async (id: string): Promise<void> => {
      const db = await initDB();
      return new Promise((resolve, reject) => {
        const transaction = db.transaction([STORE_NAME], 'readwrite');
        const store = transaction.objectStore(STORE_NAME);
        const getRequest = store.get(id);

        getRequest.onsuccess = () => {
          const report = getRequest.result as PendingReport;
          if (report) {
            report.retryCount += 1;
            const putRequest = store.put(report);
            putRequest.onsuccess = () => resolve();
            putRequest.onerror = () => reject(putRequest.error);
          } else {
            resolve();
          }
        };
        getRequest.onerror = () => reject(getRequest.error);
      });
    },
    [initDB]
  );

  // Synchroniser un rapport avec le serveur
  const syncReport = useCallback(
    async <T = unknown>(report: PendingReport, submitFn: (prevState: T, formData: FormData) => Promise<{ error?: string; success?: boolean }>): Promise<boolean> => {
      try {
        const formData = new FormData();
        formData.append('siteId', report.siteId);
        formData.append('taskId', report.taskId);
        formData.append('email', report.email);
        formData.append('name', report.name);
        formData.append('role', report.role);
        formData.append('description', report.description);
        if (report.markDone) {
          formData.append('mark_done', 'on');
        }
        if (report.photo) {
          formData.append('photo', report.photo);
        }

        const result = await submitFn({} as T, formData);

        if (result.success) {
          await removePendingReport(report.id);
          return true;
        } else {
          // Incrémenter le compteur de retry
          await incrementRetryCount(report.id);
          return false;
        }
      } catch (error) {
        console.error('Erreur lors de la synchronisation du rapport:', error);
        await incrementRetryCount(report.id);
        return false;
      }
    },
    [removePendingReport, incrementRetryCount]
  );

  // Synchroniser tous les rapports en attente
  const syncAllPendingReports = useCallback(
    async <T = unknown>(submitFn: (prevState: T, formData: FormData) => Promise<{ error?: string; success?: boolean }>): Promise<void> => {
      if (!isOnline || isSyncing) return;

      setIsSyncing(true);
      try {
        const pendingReports = await getPendingReports();
        
        for (const report of pendingReports) {
          // Limiter à 5 tentatives par rapport
          if (report.retryCount >= 5) {
            console.warn(`Rapport ${report.id} a atteint le maximum de tentatives, suppression...`);
            await removePendingReport(report.id);
            continue;
          }

          const success = await syncReport(report, submitFn);
          if (!success) {
            // Attendre un peu avant la prochaine tentative
            await new Promise((resolve) => setTimeout(resolve, 2000));
          }
        }
      } catch (error) {
        console.error('Erreur lors de la synchronisation:', error);
      } finally {
        setIsSyncing(false);
      }
    },
    [isOnline, isSyncing, getPendingReports, syncReport, removePendingReport]
  );

  // Surveiller l'état de la connexion
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
    };

    const handleOffline = () => {
      setIsOnline(false);
    };

    setIsOnline(navigator.onLine);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Mettre à jour le compteur de rapports en attente
  useEffect(() => {
    const updatePendingCount = async () => {
      try {
        const reports = await getPendingReports();
        setPendingCount(reports.length);
      } catch (error) {
        console.error('Erreur lors de la récupération du compteur:', error);
      }
    };

    updatePendingCount();
    const interval = setInterval(updatePendingCount, 5000);
    return () => clearInterval(interval);
  }, [getPendingReports]);

  // Synchroniser automatiquement quand on revient en ligne
  useEffect(() => {
    if (isOnline && pendingCount > 0) {
      // Attendre un peu pour s'assurer que la connexion est stable
      const timeout = setTimeout(() => {
        // La synchronisation sera déclenchée manuellement ou via un événement
      }, 1000);
      return () => clearTimeout(timeout);
    }
  }, [isOnline, pendingCount]);

  return {
    isOnline,
    pendingCount,
    isSyncing,
    savePendingReport,
    getPendingReports,
    removePendingReport,
    syncAllPendingReports,
  };
}

