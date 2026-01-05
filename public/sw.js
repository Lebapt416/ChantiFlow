// Service Worker pour PWA - Production Grade avec invalidation de cache
const CACHE_VERSION = 'v4'; // Incrémenter pour forcer l'invalidation (v4 pour fix CSS)
const CACHE_NAME = `chantiflow-${CACHE_VERSION}`;
const STATIC_CACHE = `chantiflow-static-${CACHE_VERSION}`;
const DYNAMIC_CACHE = `chantiflow-dynamic-${CACHE_VERSION}`;

// Ressources statiques à mettre en cache immédiatement
const STATIC_URLS = [
  '/',
  '/login',
  '/home',
  '/dashboard',
  '/manifest.json',
];

// Types de ressources à mettre en cache dynamiquement
const CACHEABLE_TYPES = [
  'text/html',
  'text/css',
  'application/javascript',
  'image/png',
  'image/jpg',
  'image/jpeg',
  'image/svg+xml',
  'application/json',
];

// Installation du service worker
self.addEventListener('install', (event) => {
  console.log('[SW] Installation du service worker', CACHE_VERSION);
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[SW] Mise en cache des ressources statiques');
        return cache.addAll(STATIC_URLS.map(url => new Request(url, { cache: 'reload' })));
      })
      .catch((error) => {
        console.error('[SW] Erreur lors de la mise en cache statique:', error);
      })
  );
  // Forcer l'activation immédiate
  self.skipWaiting();
});

// Activation du service worker avec invalidation de l'ancien cache
self.addEventListener('activate', (event) => {
  console.log('[SW] Activation du service worker', CACHE_VERSION);
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      // Supprimer TOUS les anciens caches (même ceux qui ne correspondent pas au pattern)
      const deletePromises = cacheNames.map((cacheName) => {
        // Garder uniquement les caches de la version actuelle
        if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE && cacheName !== CACHE_NAME) {
          console.log('[SW] 🧹 Suppression de l\'ancien cache:', cacheName);
          return caches.delete(cacheName);
        }
        return Promise.resolve();
      });
      
      return Promise.all(deletePromises);
    }).then(() => {
      // Prendre le contrôle immédiatement et forcer le rafraîchissement
      return self.clients.claim();
    })
  );
});

// Stratégie de cache intelligente
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Ignorer les requêtes non-GET et les requêtes vers des APIs externes
  if (request.method !== 'GET') {
    return;
  }

  // Ignorer les requêtes vers Supabase Storage (trop volumineuses)
  if (url.hostname.includes('supabase.co') && url.pathname.includes('/storage/')) {
    return;
  }

  // CRITIQUE : Ne JAMAIS mettre en cache les CSS/JS de Next.js
  // Ils doivent toujours être servis depuis le réseau pour éviter les problèmes de styles
  if (url.pathname.startsWith('/_next/static/css/') || 
      url.pathname.startsWith('/_next/static/chunks/') ||
      url.pathname.includes('.css') ||
      url.pathname.includes('.js')) {
    // Toujours servir depuis le réseau, ne jamais utiliser le cache
    return fetch(request).catch(() => {
      // En cas d'erreur réseau, ne pas utiliser le cache
      return new Response('Ressource non disponible', { status: 503 });
    });
  }

  // Stratégie : Cache First pour les ressources statiques
  if (STATIC_URLS.some(staticUrl => url.pathname === staticUrl)) {
    event.respondWith(
      caches.match(request)
        .then((cachedResponse) => {
          if (cachedResponse) {
            return cachedResponse;
          }
          return fetch(request).then((response) => {
            if (response && response.status === 200) {
              const responseToCache = response.clone();
              caches.open(STATIC_CACHE).then((cache) => {
                cache.put(request, responseToCache);
              });
            }
            return response;
          });
        })
    );
    return;
  }

  // Stratégie : Network First avec fallback pour les autres ressources
  event.respondWith(
    fetch(request)
      .then((response) => {
        // Vérifier si la réponse est valide et cacheable
        if (response && response.status === 200 && CACHEABLE_TYPES.some(type => response.headers.get('content-type')?.includes(type))) {
          const responseToCache = response.clone();
          caches.open(DYNAMIC_CACHE).then((cache) => {
            cache.put(request, responseToCache);
          });
        }
        return response;
      })
      .catch(() => {
        // En cas d'erreur réseau, chercher dans le cache
        return caches.match(request).then((cachedResponse) => {
          if (cachedResponse) {
            return cachedResponse;
          }
          // Si pas de cache, retourner une page offline basique pour les pages HTML
          if (request.headers.get('accept')?.includes('text/html')) {
            return caches.match('/').then((offlinePage) => {
              return offlinePage || new Response('Mode hors-ligne - Veuillez vous reconnecter', {
                status: 503,
                statusText: 'Service Unavailable',
                headers: { 'Content-Type': 'text/html' },
              });
            });
          }
          return new Response('Ressource non disponible hors-ligne', {
            status: 503,
            statusText: 'Service Unavailable',
          });
        });
      })
  );
});

// Gestion des messages depuis l'application
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  // Nouveau : Message pour forcer l'invalidation du cache
  if (event.data && event.data.type === 'CLEAR_CACHE') {
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => caches.delete(cacheName))
      );
    }).then(() => {
      console.log('[SW] 🧹 Cache complètement vidé');
    });
  }
});
