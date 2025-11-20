'use client';

import { QRCodeSVG } from 'qrcode.react';

type Props = {
  siteName: string;
  targetUrl: string;
};

export function SiteQrCard({ siteName, targetUrl }: Props) {
  return (
    <div className="rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900">
      <p className="text-xs uppercase tracking-[0.3em] text-zinc-500 dark:text-zinc-400">
        QR code
      </p>
      <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
        {siteName}
      </h3>
      <div className="mt-4 flex items-center justify-center rounded-2xl border border-dashed border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-950">
        <QRCodeSVG value={targetUrl} size={164} />
      </div>
      <p className="mt-4 break-all text-xs text-zinc-500 dark:text-zinc-400">
        {targetUrl}
      </p>
    </div>
  );
}

