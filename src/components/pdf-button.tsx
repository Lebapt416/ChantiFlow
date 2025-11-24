'use client';

import { FileText, Lock } from 'lucide-react';
import Link from 'next/link';

type PDFButtonProps = {
  siteId: string;
  isPro: boolean;
};

export function PDFButton({ siteId, isPro }: PDFButtonProps) {
  const handleExport = () => {
    if (!isPro) {
      // Rediriger vers la page de compte pour upgrade
      window.location.href = '/account';
      return;
    }

    // Pour l'instant, juste un alert (la génération PDF sera implémentée plus tard)
    alert('Fonctionnalité Pro - Export PDF en cours de développement');
    console.log('Export PDF pour le chantier:', siteId);
  };

  if (!isPro) {
    return (
      <button
        onClick={handleExport}
        className="flex items-center gap-2 rounded-lg border border-zinc-300 bg-zinc-100 px-4 py-2 text-sm font-semibold text-zinc-500 transition hover:bg-zinc-200 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
        title="Fonctionnalité Pro requise"
      >
        <Lock className="h-4 w-4" />
        <span>Exporter PDF</span>
      </button>
    );
  }

  return (
    <button
      onClick={handleExport}
      className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-700 active:scale-95"
    >
      <FileText className="h-4 w-4" />
      <span>📄 Exporter le rapport PDF</span>
    </button>
  );
}

