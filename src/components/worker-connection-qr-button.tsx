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
        className="inline-flex items-center gap-1.5 rounded border border-rule-soft px-3 py-1.5 text-xs font-semibold text-ink-2 transition hover:border-orange hover:text-orange"
      >
        <QrCode className="h-3.5 w-3.5" />
        QR de connexion
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-sm rounded border border-rule-soft bg-paper p-6">
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
                className="rounded p-1 text-ink-3 hover:bg-paper-2 hover:text-ink"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="mt-4 flex flex-col items-center gap-3 rounded border border-dashed border-rule-soft bg-paper-2 p-4">
              <QRCodeSVG value={loginUrl} size={180} />
              <p className="text-center text-xs text-zinc-500 dark:text-zinc-400">
                Scannez depuis le smartphone de l&apos;ouvrier pour le connecter à l&apos;app.
              </p>
            </div>

            <div className="mt-4 rounded bg-ink p-3 text-xs text-paper">
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
              className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded bg-ink px-3 py-2 text-sm font-semibold text-paper transition hover:bg-rule"
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

