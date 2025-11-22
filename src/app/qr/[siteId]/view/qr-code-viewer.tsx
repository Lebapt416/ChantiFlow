'use client';

import { QRCodeSVG } from 'qrcode.react';

type Props = {
  url: string;
  siteName: string;
};

export function QRCodeViewer({ url, siteName }: Props) {
  return (
    <div className="flex flex-col items-center">
      <div className="rounded-2xl border-2 border-zinc-200 bg-white p-6 dark:border-zinc-700 dark:bg-zinc-800 print:border-2 print:border-black">
        <div className="mb-4 text-center">
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-white print:text-black">
            {siteName}
          </h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 print:text-black">
            Scannez pour accéder au chantier
          </p>
        </div>
        <div className="flex items-center justify-center rounded-xl border border-dashed border-zinc-300 bg-zinc-50 p-6 dark:border-zinc-600 dark:bg-zinc-900 print:border-black print:bg-white">
          <QRCodeSVG
            value={url}
            size={256}
            level="H"
            includeMargin={true}
            className="print:border-0"
          />
        </div>
        <p className="mt-4 break-all text-center text-xs text-zinc-500 dark:text-zinc-400 print:text-black print:text-[10px]">
          {url}
        </p>
      </div>
    </div>
  );
}

