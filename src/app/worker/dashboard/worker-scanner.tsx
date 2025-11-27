'use client';

import dynamic from 'next/dynamic';
import { useState, useTransition } from 'react';
import { Loader2, QrCode, Scan } from 'lucide-react';
import { joinSiteAction } from './actions';

const WorkerQrScanner = dynamic(async () => {
  const mod = await import('@yudiel/react-qr-scanner');
  return mod.Scanner;
}, { ssr: false });

export function WorkerScanner() {
  const [isScanning, setIsScanning] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  const handleScan = (value: string) => {
    startTransition(async () => {
      setError(null);
      setMessage(null);
      const result = await joinSiteAction(value);
      if (result.success && result.siteId) {
        setMessage(`Chantier détecté: ${result.siteName ?? result.siteId}`);
        setTimeout(() => {
          window.location.reload();
        }, 1200);
      } else if (result.error) {
        setError(result.error);
      } else {
        setError('Scan invalide. Réessayez.');
      }
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
          onClick={() => setIsScanning((prev) => !prev)}
          className="inline-flex items-center gap-2 rounded-full bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-700 active:scale-95"
        >
          <Scan className="h-4 w-4" />
          {isScanning ? 'Arrêter' : 'Lancer'}
        </button>
      </div>

      {isScanning && (
        <div className="mt-6 overflow-hidden rounded-2xl border border-dashed border-emerald-200 bg-emerald-50/50 p-4 dark:border-emerald-800 dark:bg-emerald-900/10">
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-emerald-700 dark:text-emerald-300">
            <QrCode className="h-4 w-4" />
            Caméra active
          </div>
          <div className="aspect-square w-full max-w-sm">
            <WorkerQrScanner
              constraints={{ facingMode: 'environment' }}
              onDecode={(result) => {
                if (result) {
                  setIsScanning(false);
                  handleScan(result);
                }
              }}
              onError={(err) => {
                if (err) {
                  console.debug('QR error', err);
                }
              }}
              containerStyle={{ width: '100%', height: '100%' }}
              videoStyle={{ width: '100%', height: '100%', objectFit: 'cover' }}
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

