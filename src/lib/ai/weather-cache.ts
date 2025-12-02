/**
 * Cache simple pour les appels API météo
 * TTL: 1 heure par localisation
 */

interface WeatherCacheEntry {
  data: unknown;
  timestamp: number;
  location: string;
}

class WeatherCache {
  private cache: Map<string, WeatherCacheEntry> = new Map();
  private readonly TTL_MS = 60 * 60 * 1000; // 1 heure en millisecondes

  /**
   * Génère une clé de cache basée sur la localisation
   */
  private getCacheKey(location: string): string {
    return `weather:${location.toLowerCase().trim()}`;
  }

  /**
   * Vérifie si une entrée existe dans le cache et si elle est encore valide
   */
  get(location: string): unknown | null {
    const key = this.getCacheKey(location);
    const entry = this.cache.get(key);

    if (!entry) {
      return null;
    }

    const now = Date.now();
    const age = now - entry.timestamp;

    // Si l'entrée est expirée, la supprimer et retourner null
    if (age > this.TTL_MS) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  /**
   * Stocke une entrée dans le cache
   */
  set(location: string, data: unknown): void {
    const key = this.getCacheKey(location);
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      location: location.toLowerCase().trim(),
    });
  }

  /**
   * Supprime une entrée du cache
   */
  delete(location: string): void {
    const key = this.getCacheKey(location);
    this.cache.delete(key);
  }

  /**
   * Nettoie toutes les entrées expirées
   */
  cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      const age = now - entry.timestamp;
      if (age > this.TTL_MS) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Vide complètement le cache
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Retourne le nombre d'entrées dans le cache
   */
  size(): number {
    return this.cache.size;
  }
}

// Instance singleton
export const weatherCache = new WeatherCache();

// Nettoyage automatique toutes les heures
if (typeof setInterval !== 'undefined') {
  setInterval(() => {
    weatherCache.cleanup();
  }, 60 * 60 * 1000); // Toutes les heures
}

