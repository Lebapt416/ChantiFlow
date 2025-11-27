'use client';

import { useMemo, useState } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import { Copy, QrCode, X } from 'lucide-react';

type Props = {
  token?: string | null;
  workerName?: string | null;
  baseUrl: string;
};

export function WorkerConnectionQrButton({ token, workerName, baseUrl }: Props) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const loginUrl = useMemo(() => {
    if (!token) return null;
    const cleanBase = baseUrl.replace(/\/$/, '');
    return `${cleanBase}/worker/login?token=${token}`;
  }, [baseUrl, token]);

  if (!token || !loginUrl) {
    return null;
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-1.5 rounded-full border border-zinc-200 px-3 py-1.5 text-xs font-semibold text-zinc-700 transition hover:border-emerald-500 hover:text-emerald-600 dark:border-zinc-700 dark:text-zinc-200 dark:hover:border-emerald-400 dark:hover:text-emerald-300"
      >
        <QrCode className="h-3.5 w-3.5" />
        QR de connexion
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-sm rounded-3xl border border-zinc-200 bg-white p-6 shadow-2xl dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">Connexion</p>
                <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
                  {workerName || 'Ouvrier'}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-full p-1 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="mt-4 flex flex-col items-center gap-3 rounded-2xl border border-dashed border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900/50">
              <QRCodeSVG value={loginUrl} size={180} />
              <p className="text-center text-xs text-zinc-500 dark:text-zinc-400">
                Scannez depuis le smartphone de l&apos;ouvrier pour le connecter à l&apos;app.
              </p>
            </div>

            <div className="mt-4 rounded-2xl bg-zinc-900/90 p-3 text-xs text-white dark:bg-white/10 dark:text-zinc-100">
              <p className="font-mono break-all">{loginUrl}</p>
            </div>

            <button
              type="button"
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(loginUrl);
                  setCopied(true);
                  setTimeout(() => setCopied(false), 2000);
                } catch {
                  setCopied(false);
                }
              }}
              className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-zinc-900 px-3 py-2 text-sm font-semibold text-white transition hover:bg-zinc-800 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              <Copy className="h-4 w-4" />
              {copied ? 'Lien copié !' : 'Copier le lien'}
            </button>
          </div>
        </div>
      )}
    </>
  );
}

