'use client';

import { FileText, Lock, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { generatePDFAction } from '@/app/actions/generate-pdf';

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

      // Générer le PDF côté client
      const { jsPDF } = await import('jspdf');
      const doc = new jsPDF();

      const { site, stats, tasks, workers, reports } = result.data;

      // En-tête
      doc.setFontSize(20);
      doc.text('Rapport de Chantier', 20, 20);
      doc.setFontSize(12);
      doc.text(`Chantier: ${site.name}`, 20, 30);
      doc.text(`Date: ${new Date().toLocaleDateString('fr-FR')}`, 20, 37);
      if (site.deadline) {
        doc.text(`Deadline: ${new Date(site.deadline).toLocaleDateString('fr-FR')}`, 20, 44);
      }
      doc.text(`Adresse: ${site.address}`, 20, 51);

      let yPos = 65;

      // Statistiques
      doc.setFontSize(16);
      doc.text('Statistiques', 20, yPos);
      yPos += 10;
      doc.setFontSize(11);
      doc.text(`Tâches totales: ${stats.totalTasks}`, 20, yPos);
      yPos += 7;
      doc.text(`Tâches terminées: ${stats.doneTasks}`, 20, yPos);
      yPos += 7;
      doc.text(`Tâches en attente: ${stats.pendingTasks}`, 20, yPos);
      yPos += 7;
      doc.text(`Progression: ${stats.progress}%`, 20, yPos);
      yPos += 7;
      doc.text(`Équipe: ${stats.workersCount} membre(s)`, 20, yPos);
      yPos += 7;
      doc.text(`Rapports: ${stats.reportsCount}`, 20, yPos);
      yPos += 15;

      // Tâches
      if (tasks.length > 0) {
        doc.setFontSize(16);
        doc.text('Tâches', 20, yPos);
        yPos += 10;
        doc.setFontSize(10);

        tasks.forEach((task, index) => {
          if (yPos > 270) {
            doc.addPage();
            yPos = 20;
          }
          doc.text(`${index + 1}. ${task.title}`, 25, yPos);
          yPos += 6;
          doc.text(`   Statut: ${task.status === 'done' ? 'Terminée' : 'En attente'}`, 25, yPos);
          yPos += 6;
          if (task.role) {
            doc.text(`   Rôle: ${task.role}`, 25, yPos);
            yPos += 6;
          }
          if (task.startDate && task.endDate) {
            doc.text(
              `   Planning: ${new Date(task.startDate).toLocaleDateString('fr-FR')} - ${new Date(task.endDate).toLocaleDateString('fr-FR')}`,
              25,
              yPos,
            );
            yPos += 6;
          }
          yPos += 3;
        });
        yPos += 10;
      }

      // Équipe
      if (workers.length > 0) {
        if (yPos > 250) {
          doc.addPage();
          yPos = 20;
        }
        doc.setFontSize(16);
        doc.text('Équipe', 20, yPos);
        yPos += 10;
        doc.setFontSize(10);

        workers.forEach((worker) => {
          if (yPos > 270) {
            doc.addPage();
            yPos = 20;
          }
          doc.text(`• ${worker.name}`, 25, yPos);
          yPos += 6;
          if (worker.role) {
            doc.text(`  Rôle: ${worker.role}`, 25, yPos);
            yPos += 6;
          }
          if (worker.email) {
            doc.text(`  Email: ${worker.email}`, 25, yPos);
            yPos += 6;
          }
          yPos += 3;
        });
      }

      // Télécharger le PDF
      doc.save(`rapport-${site.name.replace(/\s+/g, '-')}-${new Date().toISOString().split('T')[0]}.pdf`);
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

