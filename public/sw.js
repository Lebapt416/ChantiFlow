// Service Worker pour PWA - Production Grade
const CACHE_NAME = 'chantiflow-v2';
const STATIC_CACHE = 'chantiflow-static-v2';
const DYNAMIC_CACHE = 'chantiflow-dynamic-v2';

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
  console.log('[SW] Installation du service worker');
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

// Activation du service worker
self.addEventListener('activate', (event) => {
  console.log('[SW] Activation du service worker');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE && cacheName !== CACHE_NAME) {
            console.log('[SW] Suppression de l\'ancien cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  // Prendre le contrôle immédiatement
  return self.clients.claim();
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
});
