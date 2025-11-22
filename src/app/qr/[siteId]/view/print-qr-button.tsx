'use client';

import { Printer } from 'lucide-react';

export function PrintQrButton() {
  const handlePrint = () => {
    window.print();
  };

  return (
    <button
      onClick={handlePrint}
      className="w-full flex items-center justify-center gap-2 rounded-lg border border-zinc-300 bg-white px-4 py-3 text-sm font-semibold text-zinc-700 transition hover:bg-zinc-50 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
    >
      <Printer className="h-5 w-5" />
      Imprimer QR code
    </button>
  );
}

