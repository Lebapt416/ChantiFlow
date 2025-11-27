'use client';

import { QRCodeSVG } from 'qrcode.react';
import { useRouter } from 'next/navigation';
import { Printer, ExternalLink } from 'lucide-react';

type Props = {
  qrUrl: string;
};

export function TeamQrSection({ qrUrl }: Props) {
  const router = useRouter();

  function handlePrint() {
    window.print();
  }

  function handleTest() {
    window.open(qrUrl, '_blank');
  }

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-white mb-2">
          QR code d&apos;inscription à l&apos;équipe
        </h3>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Partagez ce QR code pour permettre aux personnes de s&apos;ajouter à votre équipe en le scannant.
        </p>
      </div>

      {/* QR Code - visible à l'écran */}
      <div className="mb-6 flex justify-center">
        <div className="rounded-2xl border-2 border-zinc-200 bg-white p-6 dark:border-zinc-700 dark:bg-zinc-900 no-print">
          <QRCodeSVG
            value={qrUrl}
            size={256}
            level="H"
            includeMargin={true}
            className="mx-auto"
          />
          <p className="mt-4 text-center text-sm font-semibold text-zinc-900 dark:text-white">
            Rejoindre l&apos;équipe
          </p>
        </div>
        {/* QR Code - visible uniquement à l'impression */}
        <div className="rounded-2xl border-2 border-black bg-white p-6 print-qr-only" style={{ display: 'none' }}>
          <QRCodeSVG
            value={qrUrl}
            size={256}
            level="H"
            includeMargin={true}
            className="mx-auto"
          />
          <p className="mt-4 text-center text-sm font-semibold text-black">
            Rejoindre l&apos;équipe
          </p>
        </div>
      </div>

      {/* URL */}
      <div className="mb-6 rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50 no-print">
        <p className="text-xs font-semibold text-zinc-500 dark:text-zinc-400 mb-2">
          URL du QR code
        </p>
        <p className="text-sm text-zinc-900 dark:text-white break-all font-mono">
          {qrUrl}
        </p>
      </div>

      {/* Boutons d'action */}
      <div className="grid gap-3 sm:grid-cols-2 no-print">
        <button
          onClick={handleTest}
          className="flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-emerald-700 active:scale-95"
        >
          <ExternalLink className="h-4 w-4" />
          <span>Test</span>
        </button>
        <button
          onClick={handlePrint}
          className="flex items-center justify-center gap-2 rounded-lg bg-zinc-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-zinc-700 active:scale-95"
        >
          <Printer className="h-4 w-4" />
          <span>Imprimer QR code</span>
        </button>
      </div>
    </div>
  );
}

