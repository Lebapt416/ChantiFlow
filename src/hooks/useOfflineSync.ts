'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { OfflineQueue, type QueuePriority } from '@/lib/offline-queue';

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
  lastModified: number;
  retryCount: number;
  version: number;
  priority: QueuePriority;
};

const DB_NAME = 'ChantiFlowOffline';
const DB_VERSION = 2; // Incrémenté pour migration
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
        
        // Migration: créer ou mettre à jour le store
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          // Nouveau store
          const objectStore = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
          objectStore.createIndex('createdAt', 'createdAt', { unique: false });
          objectStore.createIndex('siteId', 'siteId', { unique: false });
          objectStore.createIndex('lastModified', 'lastModified', { unique: false });
          objectStore.createIndex('priority', 'priority', { unique: false });
          objectStore.createIndex('retryCount', 'retryCount', { unique: false });
        } else {
          // Migration: ajouter les nouveaux index si nécessaire
          const transaction = (event.target as IDBOpenDBRequest).transaction;
          if (transaction) {
            const objectStore = transaction.objectStore(STORE_NAME);
            
            // Ajouter index lastModified si absent
            if (!objectStore.indexNames.contains('lastModified')) {
              objectStore.createIndex('lastModified', 'lastModified', { unique: false });
            }
            
            // Ajouter index priority si absent
            if (!objectStore.indexNames.contains('priority')) {
              objectStore.createIndex('priority', 'priority', { unique: false });
            }
            
            // Ajouter index retryCount si absent
            if (!objectStore.indexNames.contains('retryCount')) {
              objectStore.createIndex('retryCount', 'retryCount', { unique: false });
            }
          }
        }
      };
    });
  }, []);

  // Sauvegarder un rapport en attente avec gestion de conflits
  const savePendingReport = useCallback(
    async (
      report: Omit<PendingReport, 'id' | 'createdAt' | 'lastModified' | 'retryCount' | 'version' | 'priority'>,
      priority: QueuePriority = 'medium',
    ): Promise<string> => {
      const db = await initDB();
      const now = Date.now();
      const id = crypto.randomUUID();
      
      const pendingReport: PendingReport = {
        ...report,
        id,
        createdAt: now,
        lastModified: now,
        retryCount: 0,
        version: 1,
        priority,
      };

      return new Promise((resolve, reject) => {
        const transaction = db.transaction([STORE_NAME], 'readwrite');
        const store = transaction.objectStore(STORE_NAME);
        
        // Vérifier si un rapport similaire existe déjà (même siteId + taskId + email)
        const index = store.index('siteId');
        const getRequest = index.getAll(report.siteId);
        
        getRequest.onsuccess = () => {
          const existing = (getRequest.result as PendingReport[]).find(
            (r) => r.taskId === report.taskId && r.email === report.email,
          );
          
          if (existing) {
            // Résolution de conflit: Last Write Wins
            if (now > existing.lastModified) {
              // Mettre à jour l'existant
              const updated: PendingReport = {
                ...existing,
                ...report,
                lastModified: now,
                version: existing.version + 1,
                priority,
              };
              
              const putRequest = store.put(updated);
              putRequest.onsuccess = () => {
                setPendingCount((prev) => {
                  // Ne pas incrémenter si c'était une mise à jour
                  return prev;
                });
                resolve(existing.id);
              };
              putRequest.onerror = () => reject(putRequest.error);
            } else {
              // Version existante plus récente, garder l'existant
              resolve(existing.id);
            }
          } else {
            // Nouveau rapport
            const addRequest = store.add(pendingReport);
            addRequest.onsuccess = () => {
              setPendingCount((prev) => prev + 1);
              resolve(id);
            };
            addRequest.onerror = () => reject(addRequest.error);
          }
        };
        
        getRequest.onerror = () => reject(getRequest.error);
      });
    },
    [initDB]
  );

  // Récupérer tous les rapports en attente, triés par priorité puis date
  const getPendingReports = useCallback(async (): Promise<PendingReport[]> => {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onsuccess = () => {
        const reports = request.result as PendingReport[];
        
        // Trier par priorité (high > medium > low) puis par date de création
        const priorityWeights: Record<QueuePriority, number> = {
          high: 3,
          medium: 2,
          low: 1,
        };
        
        const sorted = reports.sort((a, b) => {
          const priorityDiff = priorityWeights[b.priority] - priorityWeights[a.priority];
          if (priorityDiff !== 0) {
            return priorityDiff;
          }
          return a.createdAt - b.createdAt;
        });
        
        resolve(sorted);
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

  // Mettre à jour le compteur de retry avec timestamp
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
            report.lastModified = Date.now();
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

  // Synchroniser tous les rapports en attente avec queue et retry logic
  const syncAllPendingReports = useCallback(
    async <T = unknown>(submitFn: (prevState: T, formData: FormData) => Promise<{ error?: string; success?: boolean }>): Promise<void> => {
      if (!isOnline || isSyncing) return;

      setIsSyncing(true);
      try {
        const pendingReports = await getPendingReports();
        
        // Créer une queue avec les rapports
        const queue = new OfflineQueue<PendingReport>({
          maxRetries: 5,
          baseDelayMs: 1000,
          maxDelayMs: 30000,
        });
        
        // Ajouter tous les rapports à la queue
        for (const report of pendingReports) {
          queue.enqueue(report.id, report, report.priority, report.version);
        }
        
        // Traiter la queue
        let processed = 0;
        const maxConcurrent = 3; // Traiter max 3 rapports en parallèle
        
        while (queue.size() > 0 && processed < maxConcurrent) {
          const item = queue.dequeue();
          if (!item) break;
          
          // Vérifier si on peut retenter
          if (item.retryCount >= 5) {
            console.warn(`Rapport ${item.id} a atteint le maximum de tentatives, suppression...`);
            await removePendingReport(item.id);
            continue;
          }
          
          // Vérifier le backoff
          if (!queue.canRetry(item.id) && item.retryCount > 0) {
            // Remettre dans la queue pour plus tard
            queue.enqueue(item.id, item.data, item.priority, item.version);
            continue;
          }
          
          // Marquer comme en traitement
          queue.markProcessing(item.id);
          
          // Synchroniser
          const success = await syncReport(item.data, submitFn);
          
          if (success) {
            // Succès: retirer de la queue
            queue.remove(item.id);
          } else {
            // Échec: remettre dans la queue avec priorité réduite si trop d'échecs
            if (item.retryCount >= 3 && item.priority === 'high') {
              queue.enqueue(item.id, item.data, 'medium', item.version);
            } else {
              queue.enqueue(item.id, item.data, item.priority, item.version);
            }
          }
          
          processed++;
          
          // Petit délai entre chaque rapport pour éviter la surcharge
          if (processed < maxConcurrent) {
            await new Promise((resolve) => setTimeout(resolve, 500));
          }
        }
        
        // Nettoyer les éléments qui ont dépassé le max de retries
        const removedIds = queue.cleanup();
        for (const id of removedIds) {
          await removePendingReport(id);
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

/**
 * Résout un conflit entre deux versions d'un rapport (Last Write Wins)
 */
export function resolveConflict(
  local: PendingReport,
  server: PendingReport,
): PendingReport {
  // Si la version serveur est plus récente, utiliser celle-ci
  if (server.lastModified > local.lastModified || server.version > local.version) {
    return server;
  }
  // Sinon, utiliser la version locale
  return local;
}

