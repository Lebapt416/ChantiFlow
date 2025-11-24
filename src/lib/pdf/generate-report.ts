import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

// Extension de type pour jspdf-autotable
interface jsPDFCustom extends jsPDF {
  lastAutoTable: { finalY: number };
}

export async function generateSiteReport(
  site: any,
  tasks: any[],
  workers: any[],
  aiSummary: { summary: string; status: string } | null,
) {
  const doc = new jsPDF() as jsPDFCustom;
  const pageWidth = doc.internal.pageSize.width;
  const pageHeight = doc.internal.pageSize.height;
  const margin = 20;
  let currentY = 0;

  // --- COULEURS & FONTS ---
  const primaryColor = '#10b981'; // Emerald 500
  const secondaryColor = '#3f3f46'; // Zinc 700
  const lightBg = '#f4f4f5'; // Zinc 100

  // --- HEADER (Bandeau) ---
  doc.setFillColor(primaryColor);
  doc.rect(0, 0, pageWidth, 40, 'F');

  doc.setTextColor(255, 255, 255);
  doc.setFontSize(24);
  doc.setFont('helvetica', 'bold');
  doc.text('Rapport de Chantier', margin, 20);

  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  const dateStr = new Date().toLocaleDateString('fr-FR', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
  doc.text(`Généré le ${dateStr}`, margin, 30);

  // Logo / Marque à droite
  doc.setFontSize(16);
  doc.setFont('helvetica', 'bold');
  doc.text('ChantiFlow', pageWidth - margin, 20, { align: 'right' });
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text('Gestion intelligente BTP', pageWidth - margin, 28, { align: 'right' });

  currentY = 55;

  // --- INFOS CHANTIER ---
  doc.setTextColor(0, 0, 0);
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text(`Chantier : ${site.name}`, margin, currentY);

  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(100);
  currentY += 7;
  const deadline = site.deadline
    ? new Date(site.deadline).toLocaleDateString('fr-FR')
    : 'Non définie';
  doc.text(`Deadline prévue : ${deadline}`, margin, currentY);

  // Progression (Barre visuelle)
  const totalTasks = tasks.length;
  const doneTasks = tasks.filter((t) => t.status === 'done').length;
  const progress = totalTasks > 0 ? Math.round((doneTasks / totalTasks) * 100) : 0;

  doc.text(`Progression : ${progress}%`, pageWidth - margin - 50, currentY - 7);
  // Fond barre
  doc.setFillColor(220, 220, 220);
  doc.roundedRect(pageWidth - margin - 50, currentY - 4, 50, 4, 1, 1, 'F');
  // Remplissage barre
  doc.setFillColor(primaryColor);
  if (progress > 0) {
    doc.roundedRect(
      pageWidth - margin - 50,
      currentY - 4,
      50 * (progress / 100),
      4,
      1,
      1,
      'F',
    );
  }

  currentY += 15;

  // --- RÉSUMÉ IA (Encadré) ---
  if (aiSummary) {
    doc.setFillColor(lightBg);
    doc.setDrawColor(primaryColor);
    doc.roundedRect(margin, currentY, pageWidth - margin * 2, 35, 2, 2, 'FD');

    doc.setFontSize(11);
    doc.setTextColor(primaryColor);
    doc.setFont('helvetica', 'bold');
    doc.text('✨ Analyse Intelligente', margin + 5, currentY + 8);

    doc.setFontSize(10);
    doc.setTextColor(0);
    doc.setFont('helvetica', 'italic');

    // Gestion du texte long (multiligne)
    const splitText = doc.splitTextToSize(aiSummary.summary, pageWidth - margin * 2 - 10);
    doc.text(splitText, margin + 5, currentY + 16);

    currentY += 45;
  } else {
    currentY += 10;
  }

  // --- TABLEAU DES TÂCHES ---
  doc.setFontSize(12);
  doc.setTextColor(primaryColor);
  doc.setFont('helvetica', 'bold');
  doc.text('État des Tâches', margin, currentY);
  currentY += 5;

  const tasksData = tasks.map((t) => [
    t.title || 'Tâche sans titre',
    t.assignedWorkerName || 'Non assigné',
    t.status === 'done' ? 'Terminé' : 'En cours',
    t.duration_hours ? `${t.duration_hours}h` : '-',
  ]);

  autoTable(doc, {
    startY: currentY,
    head: [['Tâche', 'Responsable', 'Statut', 'Durée']],
    body: tasksData,
    theme: 'grid',
    headStyles: { fillColor: primaryColor, textColor: 255, fontStyle: 'bold' },
    styles: { fontSize: 9, cellPadding: 4 },
    alternateRowStyles: { fillColor: [245, 245, 245] },
    margin: { left: margin, right: margin },
  });

  currentY = doc.lastAutoTable.finalY + 15;

  // --- TABLEAU DE L'ÉQUIPE (Si de la place) ---
  if (currentY + 40 < pageHeight) {
    doc.setFontSize(12);
    doc.setTextColor(primaryColor);
    doc.text('Équipe sur site', margin, currentY);
    currentY += 5;

    const workersData = workers.map((w) => [
      w.name || 'Inconnu',
      w.role || 'Polyvalent',
      w.email || '-',
    ]);

    autoTable(doc, {
      startY: currentY,
      head: [['Nom', 'Rôle', 'Contact']],
      body: workersData,
      theme: 'striped',
      headStyles: { fillColor: secondaryColor },
      styles: { fontSize: 9 },
      margin: { left: margin, right: margin },
    });
  }

  // --- FOOTER (Sur toutes les pages) ---
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.setTextColor(150);
    doc.text(`ChantiFlow - Rapport généré automatiquement`, margin, pageHeight - 10);
    doc.text(`Page ${i} / ${pageCount}`, pageWidth - margin, pageHeight - 10, {
      align: 'right',
    });
  }

  // --- SAUVEGARDE ---
  const filename = `Rapport_${site.name.replace(/[^a-z0-9]/gi, '_')}_${new Date().toISOString().split('T')[0]}.pdf`;
  doc.save(filename);
}

