'use client';

import dynamic from 'next/dynamic';
import { useEffect, useState, useTransition } from 'react';
import { Loader2, QrCode, Scan } from 'lucide-react';
import { joinSiteAction } from './actions';

const WorkerQrReader = dynamic(async () => {
  const mod = await import('react-qr-reader');
  return mod.QrReader || mod.default;
}, { ssr: false });

export function WorkerScanner() {
  const [isScanning, setIsScanning] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const granted = window.localStorage.getItem('worker_camera_permission');
    if (granted === 'granted') {
      const timeout = window.setTimeout(() => setIsScanning(true), 0);
      return () => window.clearTimeout(timeout);
    }
    return undefined;
  }, []);

  const handleScan = (value: string) => {
    startTransition(async () => {
      setError(null);
      setMessage(null);
      const result = await joinSiteAction(value);
      if (result.success && result.siteId) {
        setMessage(`Chantier détecté: ${result.siteName ?? result.siteId}`);
        setTimeout(() => {
          router.push('/worker/dashboard');
        }, 1200);
      } else if (result.error) {
        setError(result.error);
      } else {
        setError('Scan invalide. Réessayez.');
      }
    });
  };

  const toggleScanner = () => {
    setIsScanning((prev) => {
      const next = !prev;
      if (next && typeof window !== 'undefined') {
        window.localStorage.setItem('worker_camera_permission', 'granted');
      }
      return next;
    });
  };

  return (
    <div className="rounded-3xl border border-zinc-200 bg-white p-6 shadow-lg dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">Connexion rapide</p>
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">Scanner un chantier</h2>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Pointez votre caméra vers le QR code posé sur le chantier pour être automatiquement assigné.
          </p>
        </div>
        <button
          type="button"
          onClick={toggleScanner}
          className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold text-white transition active:scale-95 ${
            isScanning ? 'bg-rose-600 hover:bg-rose-700' : 'bg-emerald-600 hover:bg-emerald-700'
          }`}
        >
          <Scan className="h-4 w-4" />
          {isScanning ? 'Arrêter' : 'Lancer'}
        </button>
      </div>

      {isScanning && (
        <div className="mt-6 overflow-hidden rounded-2xl border border-dashed border-emerald-200 bg-emerald-50/50 p-4 dark:border-emerald-800 dark:bg-emerald-900/10">
          <div className="mb-2 flex items-center justify-between text-sm font-semibold text-emerald-700 dark:text-emerald-300">
            <span className="flex items-center gap-2">
              <QrCode className="h-4 w-4" />
              Caméra active
            </span>
            <span className="text-xs text-emerald-500 dark:text-emerald-200">Ajustez le QR au centre du cadre</span>
          </div>
          <div className="w-full max-w-sm rounded-2xl">
            <WorkerQrReader
              constraints={{ facingMode: 'environment' }}
              scanDelay={700}
              containerStyle={{ width: '100%', height: '100%', paddingTop: 0 }}
              videoContainerStyle={{ width: '100%', height: '320px', borderRadius: '1.5rem', overflow: 'hidden' }}
              videoStyle={{ width: '100%', height: '100%', objectFit: 'cover' }}
              onResult={(result, err) => {
                if (result?.getText()) {
                  setIsScanning(false);
                  handleScan(result.getText());
                }
                if (err) {
                  console.debug('QR error', err);
                }
              }}
            />
          </div>
        </div>
      )}

      <div className="mt-4">
        <button
          type="button"
          disabled={isPending}
          onClick={() => {
            const manual = window.prompt('Entrez l\'identifiant du chantier (UUID) ou collez le QR scanné:', '');
            if (manual) {
              handleScan(manual);
            }
          }}
          className="inline-flex items-center gap-2 text-sm font-semibold text-emerald-700 hover:underline dark:text-emerald-300 disabled:opacity-60"
        >
          {isPending ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Vérification...
            </>
          ) : (
            'Saisir un identifiant manuellement'
          )}
        </button>
      </div>

      {message && (
        <p className="mt-4 rounded-lg bg-emerald-50 px-3 py-2 text-sm font-medium text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-200">
          {message}
        </p>
      )}
      {error && (
        <p className="mt-4 rounded-lg bg-rose-50 px-3 py-2 text-sm font-medium text-rose-700 dark:bg-rose-900/30 dark:text-rose-200">
          {error}
        </p>
      )}
    </div>
  );
}

