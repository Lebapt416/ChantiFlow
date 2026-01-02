import { Resend } from 'resend';
import { randomUUID } from 'crypto';
import {
  getWorkerWelcomeEmailTemplate,
  getReportNotificationEmailTemplate,
  getAccountCreatedEmailTemplate,
  getSiteCompletedEmailTemplate,
  getTeamJoinConfirmationEmailTemplate,
  getTeamApprovalEmailTemplate,
  getTeamRejectionEmailTemplate,
} from './email-templates';

// Initialiser Resend seulement si la clé API est disponible
const resend = process.env.RESEND_API_KEY ? new Resend(process.env.RESEND_API_KEY) : null;
const SUPPORT_EMAIL = process.env.SUPPORT_EMAIL || 'support@chantiflow.com';

function buildWorkerWelcomePlainText({
  workerName,
  siteName,
  managerName,
  accessCode,
  reportUrl,
}: {
  workerName: string;
  siteName?: string;
  managerName?: string;
  accessCode?: string;
  reportUrl: string;
}) {
  const lines: string[] = [
    `Bonjour ${workerName},`,
  ];

  if (siteName) {
    lines.push(
      `Vous avez été assigné${
        managerName ? ` par ${managerName}` : ''
      } au chantier ${siteName}.`,
    );
  } else {
    lines.push(
      `Vous avez été ajouté${
        managerName ? ` par ${managerName}` : ''
      } à l'équipe ChantiFlow.`,
    );
  }

  if (accessCode) {
    lines.push(`Votre code d'accès : ${accessCode}`);
    lines.push(`Gardez ce code. Il vous sera demandé après avoir scanné le QR code du chantier.`);
  } else {
    lines.push("Vous recevrez un code d'accès dès que vous serez assigné à un chantier.");
  }

  lines.push(`Accédez directement au formulaire : ${reportUrl}`);
  lines.push('— Équipe ChantiFlow');

  return lines.join('\n\n');
}

/**
 * Fonction générique pour envoyer un email
 */
export async function sendEmail({
  to,
  subject,
  html,
}: {
  to: string;
  subject: string;
  html: string;
}) {
  if (!process.env.RESEND_API_KEY || !resend) {
    console.warn('RESEND_API_KEY non configuré, email non envoyé');
    return { success: false, error: 'Service email non configuré' };
  }

  try {
    const fromEmail = process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>';
    const supportEmail = process.env.SUPPORT_EMAIL || 'support@chantiflow.com';
    
    const { data, error } = await resend.emails.send({
      from: fromEmail,
      to,
      subject,
      html,
      headers: {
        'X-Entity-Ref-ID': randomUUID(),
        'List-Unsubscribe': `<mailto:${supportEmail}?subject=unsubscribe>`,
        'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click',
        'X-Priority': '1',
        'X-MSMail-Priority': 'High',
      },
      tags: [
        { name: 'category', value: 'transactional' },
        { name: 'source', value: 'chantiflow' },
      ],
    });

    if (error) {
      console.error('❌ Erreur Resend:', error);
      return { success: false, error: error.message };
    }

    console.log('✅ Email envoyé avec succès à:', to);
    return { success: true, data };
  } catch (error) {
    console.error('❌ Exception lors de l\'envoi email:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Erreur inconnue' };
  }
}

export async function sendWorkerWelcomeEmail({
  workerEmail,
  workerName,
  siteName,
  siteId,
  managerName,
  managerEmail,
  accessCode,
}: {
  workerEmail: string;
  workerName: string;
  siteName?: string;
  siteId?: string;
  managerName?: string;
  managerEmail?: string;
  accessCode?: string;
}) {
  // Si Resend n'est pas configuré, on retourne silencieusement
  if (!process.env.RESEND_API_KEY) {
    console.warn('RESEND_API_KEY non configuré, email non envoyé');
    return { success: false, error: 'Service email non configuré' };
  }

  // Vérifier que Resend est initialisé
  if (!resend) {
    console.warn('Resend non initialisé');
    return { success: false, error: 'Service email non initialisé' };
  }

  const appUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? '';
  const reportUrl = siteId ? `${appUrl}/qr/${siteId}` : `${appUrl}/reports`;

  // Log pour debug
  console.log('📧 Préparation email - accessCode:', accessCode, 'siteId:', siteId, 'type:', typeof accessCode);
  
  // Forcer l'affichage du code même s'il est undefined pour debug
  if (!accessCode) {
    console.warn('⚠️ ATTENTION: accessCode est undefined/null dans sendWorkerWelcomeEmail');
  }

  try {
    // Vérifier que Resend est configuré
    if (!process.env.RESEND_API_KEY) {
      console.warn('RESEND_API_KEY non configuré, email non envoyé');
      return { success: false, error: 'Service email non configuré' };
    }

    const normalizedAccessCode = accessCode?.trim()
      ? accessCode.trim().toUpperCase()
      : undefined;

    const plainText = buildWorkerWelcomePlainText({
      workerName,
      siteName,
      managerName,
      accessCode: normalizedAccessCode,
      reportUrl,
    });

    const fromEmail = process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>';
    const supportEmail = SUPPORT_EMAIL;
    
    const { data, error } = await resend.emails.send({
      from: fromEmail,
      to: workerEmail,
      subject: siteName
        ? `Bienvenue sur le chantier ${siteName} - ChantiFlow`
        : 'Bienvenue dans l\'équipe ChantiFlow',
      html: getWorkerWelcomeEmailTemplate({
        workerName,
        siteName,
        managerName,
        accessCode: normalizedAccessCode,
        reportUrl,
        siteId: siteId || undefined,
      }),
      text: plainText,
      replyTo: managerEmail || supportEmail,
      headers: {
        'X-Entity-Ref-ID': randomUUID(),
        'List-Unsubscribe': `<mailto:${supportEmail}?subject=unsubscribe>`,
        'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click',
        'X-Priority': '1',
        'X-MSMail-Priority': 'High',
        'Precedence': 'bulk',
      },
      tags: [
        { name: 'category', value: 'transactional' },
        { name: 'type', value: 'welcome' },
        { name: 'source', value: 'chantiflow' },
      ],
    });

    if (error) {
      console.error('❌ Erreur Resend lors de l\'envoi email:', {
        message: error.message,
        name: error.name,
        to: workerEmail,
      });
      return { success: false, error: error.message };
    }

    console.log('✅ Email de bienvenue envoyé avec succès à:', workerEmail);
    return { success: true, data };
  } catch (error) {
    console.error('❌ Exception lors de l\'envoi email:', {
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
      to: workerEmail,
    });
    return { success: false, error: error instanceof Error ? error.message : 'Erreur inconnue' };
  }
}

export async function sendReportNotificationEmail({
  managerEmail,
  managerName,
  workerName,
  workerEmail,
  taskTitle,
  siteName,
  reportUrl,
}: {
  managerEmail: string;
  managerName?: string;
  workerName: string;
  workerEmail?: string;
  taskTitle: string;
  siteName: string;
  reportUrl: string;
}) {
  // Si Resend n'est pas configuré, on retourne silencieusement
  if (!process.env.RESEND_API_KEY) {
    console.warn('RESEND_API_KEY non configuré, email non envoyé');
    return { success: false, error: 'Service email non configuré' };
  }

  // Vérifier que Resend est initialisé
  if (!resend) {
    console.warn('Resend non initialisé');
    return { success: false, error: 'Service email non initialisé' };
  }

  try {

    const fromEmail = process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>';
    const supportEmail = SUPPORT_EMAIL;
    
    const { data, error } = await resend.emails.send({
      from: fromEmail,
      to: managerEmail,
      subject: `Nouveau rapport - ${taskTitle} - ${siteName}`,
      html: getReportNotificationEmailTemplate({
        managerName,
        workerName,
        workerEmail,
        taskTitle,
        siteName,
        reportUrl,
      }),
      headers: {
        'X-Entity-Ref-ID': randomUUID(),
        'List-Unsubscribe': `<mailto:${supportEmail}?subject=unsubscribe>`,
        'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click',
        'X-Priority': '1',
        'X-MSMail-Priority': 'High',
      },
      tags: [
        { name: 'category', value: 'notification' },
        { name: 'type', value: 'report' },
        { name: 'source', value: 'chantiflow' },
      ],
    });

    if (error) {
      console.error('❌ Erreur Resend lors de l\'envoi notification:', {
        message: error.message,
        name: error.name,
        to: managerEmail,
      });
      return { success: false, error: error.message };
    }

    console.log('✅ Email de notification envoyé avec succès à:', managerEmail);
    return { success: true, data };
  } catch (error) {
    console.error('❌ Exception lors de l\'envoi notification:', {
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
      to: managerEmail,
    });
    return { success: false, error: error instanceof Error ? error.message : 'Erreur inconnue' };
  }
}

export async function sendAccountCreatedEmail({
  userEmail,
  temporaryPassword,
}: {
  userEmail: string;
  temporaryPassword: string;
}) {
  // Si Resend n'est pas configuré, on retourne silencieusement
  if (!process.env.RESEND_API_KEY) {
    console.warn('RESEND_API_KEY non configuré, email non envoyé');
    return { success: false, error: 'Service email non configuré' };
  }

  // Vérifier que Resend est initialisé
  if (!resend) {
    console.warn('Resend non initialisé');
    return { success: false, error: 'Service email non initialisé' };
  }

  const appUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? '';
  const loginUrl = `${appUrl}/login`;

  try {
    const fromEmail = process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>';
    const supportEmail = SUPPORT_EMAIL;
    
    const { data, error } = await resend.emails.send({
      from: fromEmail,
      to: userEmail,
      subject: 'Bienvenue sur ChantiFlow - Votre compte a été créé',
      html: getAccountCreatedEmailTemplate({
        userEmail,
        temporaryPassword,
        loginUrl,
      }),
      headers: {
        'X-Entity-Ref-ID': randomUUID(),
        'List-Unsubscribe': `<mailto:${supportEmail}?subject=unsubscribe>`,
        'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click',
        'X-Priority': '1',
        'X-MSMail-Priority': 'High',
      },
      tags: [
        { name: 'category', value: 'transactional' },
        { name: 'type', value: 'account-created' },
        { name: 'source', value: 'chantiflow' },
      ],
    });

    if (error) {
      console.error('❌ Erreur envoi email compte créé:', error);
      return { success: false, error: error.message };
    }

    console.log('✅ Email identifiants envoyé à', userEmail);
    return { success: true, data };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } catch (error: any) {
    console.error('❌ Exception envoi email compte créé:', error);
    return { success: false, error: error?.message || 'Erreur inconnue' };
  }
}

export async function sendSiteCompletedEmail({
  workerEmail,
  workerName,
  siteName,
}: {
  workerEmail: string;
  workerName: string;
  siteName: string;
}) {
  // Si Resend n'est pas configuré, on retourne silencieusement
  if (!process.env.RESEND_API_KEY) {
    console.warn('RESEND_API_KEY non configuré, email non envoyé');
    return { success: false, error: 'Service email non configuré' };
  }

  // Vérifier que Resend est initialisé
  if (!resend) {
    console.warn('Resend non initialisé');
    return { success: false, error: 'Service email non initialisé' };
  }

  try {
    const fromEmail = process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>';
    const supportEmail = SUPPORT_EMAIL;
    
    const { data, error } = await resend.emails.send({
      from: fromEmail,
      to: workerEmail,
      subject: `Chantier terminé - ${siteName} - ChantiFlow`,
      html: getSiteCompletedEmailTemplate({
        workerName,
        siteName,
      }),
      headers: {
        'X-Entity-Ref-ID': randomUUID(),
        'List-Unsubscribe': `<mailto:${supportEmail}?subject=unsubscribe>`,
        'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click',
        'X-Priority': '1',
        'X-MSMail-Priority': 'High',
      },
      tags: [
        { name: 'category', value: 'notification' },
        { name: 'type', value: 'site-completed' },
        { name: 'source', value: 'chantiflow' },
      ],
    });

    if (error) {
      console.error('❌ Erreur envoi email fin de chantier:', error);
      return { success: false, error: error.message };
    }

    console.log('✅ Email fin de chantier envoyé à', workerEmail);
    return { success: true, data };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } catch (error: any) {
    console.error('❌ Exception envoi email fin de chantier:', error);
    return { success: false, error: error?.message || 'Erreur inconnue' };
  }
}

export async function sendTeamJoinConfirmationEmail({
  workerEmail,
  workerName,
  managerName,
}: {
  workerEmail?: string;
  workerName: string;
  managerName?: string;
}) {
  // Si Resend n'est pas configuré, on retourne silencieusement
  if (!process.env.RESEND_API_KEY) {
    console.warn('RESEND_API_KEY non configuré, email non envoyé');
    return { success: false, error: 'Service email non configuré' };
  }

  // Vérifier que Resend est initialisé
  if (!resend) {
    console.warn('Resend non initialisé');
    return { success: false, error: 'Service email non initialisé' };
  }

  // Si pas d'email, on ne peut pas envoyer
  if (!workerEmail) {
    console.warn('Pas d\'email fourni pour l\'envoi de confirmation');
    return { success: false, error: 'Email non fourni' };
  }

  try {
    console.log('📧 Resend: Préparation envoi email à:', workerEmail);
    console.log('📧 Resend: From email:', process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>');
    
    const fromEmail = process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>';
    const supportEmail = SUPPORT_EMAIL;
    
    const { data, error } = await resend.emails.send({
      from: fromEmail,
      to: workerEmail,
      subject: 'Demande d\'ajout à l\'équipe - En attente de validation',
      html: getTeamJoinConfirmationEmailTemplate({
        workerName,
        managerName,
      }),
      headers: {
        'X-Entity-Ref-ID': randomUUID(),
        'List-Unsubscribe': `<mailto:${supportEmail}?subject=unsubscribe>`,
        'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click',
        'X-Priority': '1',
        'X-MSMail-Priority': 'High',
      },
      tags: [
        { name: 'category', value: 'transactional' },
        { name: 'type', value: 'team-join-confirmation' },
        { name: 'source', value: 'chantiflow' },
      ],
    });

    if (error) {
      console.error('❌ Erreur Resend lors de l\'envoi confirmation:', {
        message: error.message,
        name: error.name,
        to: workerEmail,
      });
      
      // Vérifier si l'erreur est due à une restriction Resend (email non autorisé en mode dev)
      if (error.message?.includes('not authorized') || 
          error.message?.includes('domain') || 
          error.message?.includes('verified') ||
          error.message?.includes('not allowed')) {
        return { 
          success: false, 
          error: 'Envoi d\'email restreint. En mode développement, Resend ne peut envoyer des emails qu\'à l\'adresse email vérifiée de votre compte. Pour envoyer à d\'autres adresses, vous devez vérifier un domaine dans Resend.' 
        };
      }
      
      return { success: false, error: error.message };
    }

    console.log('✅ Email de confirmation envoyé avec succès à:', workerEmail);
    return { success: true, data };
  } catch (error) {
    console.error('❌ Exception lors de l\'envoi confirmation:', {
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
      to: workerEmail,
    });
    return { success: false, error: error instanceof Error ? error.message : 'Erreur inconnue' };
  }
}

export async function sendTeamApprovalEmail({
  workerEmail,
  workerName,
  managerName,
}: {
  workerEmail?: string;
  workerName: string;
  managerName?: string;
}) {
  // Si Resend n'est pas configuré, on retourne silencieusement
  if (!process.env.RESEND_API_KEY) {
    console.warn('RESEND_API_KEY non configuré, email non envoyé');
    return { success: false, error: 'Service email non configuré' };
  }

  // Vérifier que Resend est initialisé
  if (!resend) {
    console.warn('Resend non initialisé');
    return { success: false, error: 'Service email non initialisé' };
  }

  // Si pas d'email, on ne peut pas envoyer
  if (!workerEmail) {
    console.warn('Pas d\'email fourni pour l\'envoi d\'approbation');
    return { success: false, error: 'Email non fourni' };
  }

  try {
    const fromEmail = process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>';
    const supportEmail = SUPPORT_EMAIL;
    
    const { data, error } = await resend.emails.send({
      from: fromEmail,
      to: workerEmail,
      subject: 'Demande d\'ajout à l\'équipe - Approuvée',
      html: getTeamApprovalEmailTemplate({
        workerName,
        managerName,
      }),
      headers: {
        'X-Entity-Ref-ID': randomUUID(),
        'List-Unsubscribe': `<mailto:${supportEmail}?subject=unsubscribe>`,
        'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click',
        'X-Priority': '1',
        'X-MSMail-Priority': 'High',
      },
      tags: [
        { name: 'category', value: 'transactional' },
        { name: 'type', value: 'team-approval' },
        { name: 'source', value: 'chantiflow' },
      ],
    });

    if (error) {
      console.error('❌ Erreur Resend lors de l\'envoi approbation:', {
        message: error.message,
        name: error.name,
        to: workerEmail,
      });
      return { success: false, error: error.message };
    }

    console.log('✅ Email d\'approbation envoyé avec succès à:', workerEmail);
    return { success: true, data };
  } catch (error) {
    console.error('❌ Exception lors de l\'envoi approbation:', {
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
      to: workerEmail,
    });
    return { success: false, error: error instanceof Error ? error.message : 'Erreur inconnue' };
  }
}

export async function sendTeamRejectionEmail({
  workerEmail,
  workerName,
  managerName,
}: {
  workerEmail?: string;
  workerName: string;
  managerName?: string;
}) {
  // Si Resend n'est pas configuré, on retourne silencieusement
  if (!process.env.RESEND_API_KEY) {
    console.warn('RESEND_API_KEY non configuré, email non envoyé');
    return { success: false, error: 'Service email non configuré' };
  }

  // Vérifier que Resend est initialisé
  if (!resend) {
    console.warn('Resend non initialisé');
    return { success: false, error: 'Service email non initialisé' };
  }

  // Si pas d'email, on ne peut pas envoyer
  if (!workerEmail) {
    console.warn('Pas d\'email fourni pour l\'envoi de refus');
    return { success: false, error: 'Email non fourni' };
  }

  try {
    const fromEmail = process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>';
    const supportEmail = SUPPORT_EMAIL;
    
    const { data, error } = await resend.emails.send({
      from: fromEmail,
      to: workerEmail,
      subject: 'Demande d\'ajout à l\'équipe - Refusée',
      html: getTeamRejectionEmailTemplate({
        workerName,
        managerName,
      }),
      headers: {
        'X-Entity-Ref-ID': randomUUID(),
        'List-Unsubscribe': `<mailto:${supportEmail}?subject=unsubscribe>`,
        'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click',
        'X-Priority': '1',
        'X-MSMail-Priority': 'High',
      },
      tags: [
        { name: 'category', value: 'transactional' },
        { name: 'type', value: 'team-rejection' },
        { name: 'source', value: 'chantiflow' },
      ],
    });

    if (error) {
      console.error('❌ Erreur Resend lors de l\'envoi refus:', {
        message: error.message,
        name: error.name,
        to: workerEmail,
      });
      return { success: false, error: error.message };
    }

    console.log('✅ Email de refus envoyé avec succès à:', workerEmail);
    return { success: true, data };
  } catch (error) {
    console.error('❌ Exception lors de l\'envoi refus:', {
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
      to: workerEmail,
    });
    return { success: false, error: error instanceof Error ? error.message : 'Erreur inconnue' };
  }
}

