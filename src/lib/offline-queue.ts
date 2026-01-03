/**
 * File d'attente robuste pour la synchronisation offline
 * Gère les priorités, retry logic avec backoff exponentiel, et échecs persistants
 */

export type QueuePriority = 'high' | 'medium' | 'low';

export type QueueItem<T = unknown> = {
  id: string;
  data: T;
  priority: QueuePriority;
  retryCount: number;
  lastRetryAt: number | null;
  createdAt: number;
  lastModified: number;
  version: number;
};

export type QueueConfig = {
  maxRetries?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  priorityWeights?: Record<QueuePriority, number>;
};

const DEFAULT_CONFIG: Required<QueueConfig> = {
  maxRetries: 5,
  baseDelayMs: 1000,
  maxDelayMs: 30000,
  priorityWeights: {
    high: 3,
    medium: 2,
    low: 1,
  },
};

/**
 * File d'attente avec support des priorités et retry logic
 */
export class OfflineQueue<T = unknown> {
  private items: Map<string, QueueItem<T>> = new Map();
  private config: Required<QueueConfig>;

  constructor(config: QueueConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Ajoute un élément à la queue
   */
  enqueue(
    id: string,
    data: T,
    priority: QueuePriority = 'medium',
    version: number = 1,
  ): void {
    const now = Date.now();
    const existing = this.items.get(id);

    if (existing) {
      // Mise à jour avec résolution de conflit (Last Write Wins)
      if (version > existing.version || now > existing.lastModified) {
        this.items.set(id, {
          ...existing,
          data,
          priority,
          version,
          lastModified: now,
        });
      }
    } else {
      // Nouvel élément
      this.items.set(id, {
        id,
        data,
        priority,
        retryCount: 0,
        lastRetryAt: null,
        createdAt: now,
        lastModified: now,
        version,
      });
    }
  }

  /**
   * Récupère le prochain élément à traiter (FIFO avec priorités)
   */
  dequeue(): QueueItem<T> | null {
    if (this.items.size === 0) {
      return null;
    }

    // Trier par priorité puis par date de création
    const sorted = Array.from(this.items.values()).sort((a, b) => {
      // Comparer les priorités
      const priorityDiff =
        this.config.priorityWeights[b.priority] -
        this.config.priorityWeights[a.priority];
      if (priorityDiff !== 0) {
        return priorityDiff;
      }
      // Si même priorité, FIFO (plus ancien en premier)
      return a.createdAt - b.createdAt;
    });

    return sorted[0] || null;
  }

  /**
   * Marque un élément comme en cours de traitement
   */
  markProcessing(id: string): void {
    const item = this.items.get(id);
    if (item) {
      item.lastRetryAt = Date.now();
      item.retryCount += 1;
      item.lastModified = Date.now();
    }
  }

  /**
   * Supprime un élément de la queue (succès)
   */
  remove(id: string): boolean {
    return this.items.delete(id);
  }

  /**
   * Vérifie si un élément peut être retenté
   */
  canRetry(id: string): boolean {
    const item = this.items.get(id);
    if (!item) {
      return false;
    }

    if (item.retryCount >= this.config.maxRetries) {
      return false;
    }

    // Calculer le délai avec backoff exponentiel
    const delay = this.calculateBackoffDelay(item.retryCount);
    const now = Date.now();
    const lastRetry = item.lastRetryAt || item.createdAt;

    return now - lastRetry >= delay;
  }

  /**
   * Calcule le délai de backoff exponentiel
   */
  private calculateBackoffDelay(retryCount: number): number {
    const delay = Math.min(
      this.config.baseDelayMs * Math.pow(2, retryCount),
      this.config.maxDelayMs,
    );
    return delay;
  }

  /**
   * Récupère tous les éléments en attente
   */
  getAll(): QueueItem<T>[] {
    return Array.from(this.items.values()).sort((a, b) => {
      const priorityDiff =
        this.config.priorityWeights[b.priority] -
        this.config.priorityWeights[a.priority];
      if (priorityDiff !== 0) {
        return priorityDiff;
      }
      return a.createdAt - b.createdAt;
    });
  }

  /**
   * Récupère un élément par ID
   */
  get(id: string): QueueItem<T> | null {
    return this.items.get(id) || null;
  }

  /**
   * Compte le nombre d'éléments
   */
  size(): number {
    return this.items.size;
  }

  /**
   * Compte par priorité
   */
  countByPriority(): Record<QueuePriority, number> {
    const counts: Record<QueuePriority, number> = {
      high: 0,
      medium: 0,
      low: 0,
    };

    for (const item of this.items.values()) {
      counts[item.priority] += 1;
    }

    return counts;
  }

  /**
   * Nettoie les éléments qui ont dépassé le max de retries
   */
  cleanup(): string[] {
    const removedIds: string[] = [];

    for (const [id, item] of this.items.entries()) {
      if (item.retryCount >= this.config.maxRetries) {
        this.items.delete(id);
        removedIds.push(id);
      }
    }

    return removedIds;
  }

  /**
   * Vide la queue
   */
  clear(): void {
    this.items.clear();
  }
}

