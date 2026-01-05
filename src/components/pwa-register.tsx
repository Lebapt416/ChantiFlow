'use client';

import { useEffect } from 'react';

export function PwaRegister() {
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker
          .register('/sw.js')
          .then((registration) => {
            console.log('[PWA] Service Worker enregistré avec succès:', registration.scope);
            
            // Vérifier s'il y a une mise à jour du Service Worker
            registration.addEventListener('updatefound', () => {
              const newWorker = registration.installing;
              if (newWorker) {
                newWorker.addEventListener('statechange', () => {
                  if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                    // Nouveau Service Worker disponible - forcer l'activation
                    console.log('[PWA] 🆕 Nouveau Service Worker disponible - Activation...');
                    newWorker.postMessage({ type: 'SKIP_WAITING' });
                    
                    // Forcer le rechargement après activation
                    window.location.reload();
                  }
                });
              }
            });
          })
          .catch((error) => {
            console.error('[PWA] Erreur lors de l\'enregistrement du Service Worker:', error);
          });
      });
    }
  }, []);

  return null;
}

