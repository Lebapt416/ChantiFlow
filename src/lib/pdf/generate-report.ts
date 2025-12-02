import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

// Extension de type pour jspdf-autotable
interface jsPDFCustom extends jsPDF {
  lastAutoTable: { finalY: number };
}

// Couleurs professionnelles
const COLORS = {
  navy: [13, 27, 62], // Bleu foncé (navy) pour les en-têtes
  lightGray: [245, 245, 247], // Gris très clair pour les lignes alternées
  white: [255, 255, 255],
  darkGray: [100, 100, 100],
  textDark: [30, 30, 30],
};

/**
 * Convertit une couleur hex en RGB
 */
function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [
        parseInt(result[1], 16),
        parseInt(result[2], 16),
        parseInt(result[3], 16),
      ]
    : [0, 0, 0];
}

/**
 * Génère un rapport de chantier professionnel en PDF
 */
export async function generateSiteReport(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  site: any,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tasks: any[],
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  workers: any[],
  aiSummary: { summary: string; status: string; sites_mentioned?: string[] } | null,
) {
  const doc = new jsPDF() as jsPDFCustom;
  const pageWidth = doc.internal.pageSize.width;
  const pageHeight = doc.internal.pageSize.height;
  const margin = 20;
  let currentY = margin;

  // ============================================
  // 1. EN-TÊTE : Logo (gauche) + Titre + Date (droite)
  // ============================================
  const headerHeight = 50;
  
  // Logo / Nom de l'entreprise à gauche
  doc.setFontSize(20);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.navy);
  doc.text('ChantiFlow', margin, currentY + 10);
  
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...COLORS.darkGray);
  doc.text('Gestion intelligente de chantiers', margin, currentY + 16);

  // Titre du rapport et date à droite
  doc.setFontSize(18);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.textDark);
  doc.text('Rapport de Chantier', pageWidth - margin, currentY + 8, { align: 'right' });
  
  const dateStr = new Date().toLocaleDateString('fr-FR', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...COLORS.darkGray);
  doc.text(`Généré le ${dateStr}`, pageWidth - margin, currentY + 16, { align: 'right' });

  // Ligne de séparation
  doc.setDrawColor(200, 200, 200);
  doc.setLineWidth(0.5);
  doc.line(margin, currentY + 22, pageWidth - margin, currentY + 22);

  currentY = currentY + headerHeight;

  // ============================================
  // 2. INFO CHANTIER : Cadre gris clair
  // ============================================
  const infoBoxHeight = 50;
  const infoBoxPadding = 10;

  // Fond gris clair
  doc.setFillColor(...COLORS.lightGray);
  doc.setDrawColor(220, 220, 220);
  doc.setLineWidth(0.5);
  doc.roundedRect(margin, currentY, pageWidth - margin * 2, infoBoxHeight, 3, 3, 'FD');

  // Contenu du cadre
  const infoStartX = margin + infoBoxPadding;
  const infoStartY = currentY + infoBoxPadding;
  let infoY = infoStartY;

  doc.setFontSize(12);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.textDark);
  doc.text('Informations du Chantier', infoStartX, infoY);
  infoY += 8;

  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...COLORS.darkGray);

  // Nom du client / Chantier
  const siteName = site.name || 'Non spécifié';
  doc.text(`Nom du chantier : ${siteName}`, infoStartX, infoY);
  infoY += 7;

  // Adresse
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const address = (site as any).address || (site as any).postal_code || (site as any).location || 'Non spécifié';
  doc.text(`Adresse : ${address}`, infoStartX, infoY);
  infoY += 7;

  // Chef de chantier (récupéré depuis l'utilisateur ou site)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const siteManager = (site as any).manager_name || (site as any).created_by_name || (site as any).manager || 'Non spécifié';
  doc.text(`Chef de chantier : ${siteManager}`, infoStartX, infoY);

  currentY = currentY + infoBoxHeight + 15;

  // ============================================
  // 3. TABLEAU DES TÂCHES : autoTable avec style professionnel
  // ============================================
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.textDark);
  doc.text('État des Tâches', margin, currentY);
  currentY += 8;

  // Préparer les données du tableau
  const tasksData = tasks.map((t) => {
    const statusText = t.status === 'done' ? 'Terminé' : t.status === 'pending' ? 'En attente' : 'En cours';
    const assignedTo = t.assignedWorkerName || 'Non assigné';
    const date = t.startDate 
      ? new Date(t.startDate).toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', year: 'numeric' })
      : '-';
    
    return [
      t.title || 'Tâche sans titre',
      assignedTo,
      statusText,
      date,
    ];
  });

  // Générer le tableau avec autoTable
  autoTable(doc, {
    startY: currentY,
    head: [['Tâche', 'Assigné à', 'Statut', 'Date']],
    body: tasksData.length > 0 ? tasksData : [['Aucune tâche', '-', '-', '-']],
    theme: 'striped',
    headStyles: {
      fillColor: COLORS.navy,
      textColor: COLORS.white,
      fontStyle: 'bold',
      fontSize: 10,
      cellPadding: 6,
    },
    bodyStyles: {
      fontSize: 9,
      cellPadding: 5,
      textColor: COLORS.textDark,
    },
    alternateRowStyles: {
      fillColor: COLORS.lightGray,
    },
    styles: {
      cellPadding: 5,
      overflow: 'linebreak',
      cellWidth: 'wrap',
    },
    columnStyles: {
      0: { cellWidth: 'auto' }, // Tâche
      1: { cellWidth: 'auto' }, // Assigné à
      2: { cellWidth: 'auto' }, // Statut
      3: { cellWidth: 'auto' }, // Date
    },
    margin: { left: margin, right: margin, top: 5 },
    didDrawPage: () => {
      // Le pied de page sera ajouté manuellement après
    },
  });

  currentY = doc.lastAutoTable.finalY + 15;

  // ============================================
  // 4. PIED DE PAGE : Numéro de page centré sur toutes les pages
  // ============================================
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pageCount = (doc as any).getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (doc as any).setPage(i);
    addFooter(doc, pageWidth, pageHeight, margin, i, pageCount);
  }

  // ============================================
  // SAUVEGARDE
  // ============================================
  const sanitizedName = site.name
    ? site.name.replace(/[^a-z0-9]/gi, '_').substring(0, 50)
    : 'chantier';
  const dateISO = new Date().toISOString().split('T')[0];
  const filename = `Rapport_${sanitizedName}_${dateISO}.pdf`;
  
  doc.save(filename);
}

/**
 * Ajoute le pied de page avec le numéro de page centré
 */
function addFooter(doc: jsPDF, pageWidth: number, pageHeight: number, margin: number, currentPage: number, pageCount: number) {
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...COLORS.darkGray);
  
  // Numéro de page centré
  const footerText = `Page ${currentPage} sur ${pageCount}`;
  const textWidth = doc.getTextWidth(footerText);
  const xPosition = (pageWidth - textWidth) / 2;
  
  doc.text(footerText, xPosition, pageHeight - margin / 2);
}

