'use client';

import { useEffect, useRef } from 'react';

/**
 * Composant de debug temporaire pour identifier les causes de rechargement
 * À supprimer après diagnostic
 */
export function AuthProviderDebug() {
  const renderCountRef = useRef(0);
  const lastPathRef = useRef<string>('');

  useEffect(() => {
    if (typeof window === 'undefined') return;

    renderCountRef.current += 1;
    const currentPath = window.location.pathname;

    console.log('[DEBUG] AuthProvider render #', renderCountRef.current);
    console.log('[DEBUG] Current path:', currentPath);
    console.log('[DEBUG] Path changed:', lastPathRef.current !== currentPath);
    console.log('[DEBUG] Stack trace:', new Error().stack);

    if (lastPathRef.current && lastPathRef.current !== currentPath) {
      console.warn('[DEBUG] ⚠️ PATH CHANGED:', lastPathRef.current, '->', currentPath);
    }

    lastPathRef.current = currentPath;

    // Détecter les rechargements de page
    const handleBeforeUnload = () => {
      console.warn('[DEBUG] 🔄 PAGE RELOAD DETECTED');
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  });

  return null;
}

