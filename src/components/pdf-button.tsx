'use client';

import { FileText, Lock, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { generatePDFAction } from '@/app/actions/generate-pdf';
import { generateSiteReport } from '@/lib/pdf/generate-report';

type PDFButtonProps = {
  siteId: string;
  isPro: boolean;
};

export function PDFButton({ siteId, isPro }: PDFButtonProps) {
  const [loading, setLoading] = useState(false);

  const handleExport = async () => {
    if (!isPro) {
      window.location.href = '/account';
      return;
    }

    setLoading(true);
    try {
      const result = await generatePDFAction(siteId);

      if (result.error) {
        alert(`Erreur: ${result.error}`);
        setLoading(false);
        return;
      }

      if (!result.data) {
        alert('Aucune donnée à exporter');
        setLoading(false);
        return;
      }

      const { site, tasks, workers, aiSummary } = result.data;

      // Utiliser la nouvelle fonction de génération professionnelle
      await generateSiteReport(site, tasks, workers, aiSummary);
    } catch (error) {
      console.error('Erreur génération PDF:', error);
      alert('Erreur lors de la génération du PDF. Vérifiez la console pour plus de détails.');
    } finally {
      setLoading(false);
    }
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
      disabled={loading}
      className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-700 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {loading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        <FileText className="h-4 w-4" />
      )}
      <span>{loading ? 'Génération...' : '📄 Exporter le rapport PDF'}</span>
    </button>
  );
}

