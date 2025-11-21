import { Resend } from 'resend';

// Initialiser Resend seulement si la clé API est disponible
const resend = process.env.RESEND_API_KEY ? new Resend(process.env.RESEND_API_KEY) : null;

export async function sendWorkerWelcomeEmail({
  workerEmail,
  workerName,
  siteName,
  siteId,
  managerName,
}: {
  workerEmail: string;
  workerName: string;
  siteName?: string;
  siteId?: string;
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

  const appUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? '';
  const reportUrl = siteId ? `${appUrl}/qr/${siteId}` : `${appUrl}/reports`;

  try {
    // Vérifier que Resend est configuré
    if (!process.env.RESEND_API_KEY) {
      console.warn('RESEND_API_KEY non configuré, email non envoyé');
      return { success: false, error: 'Service email non configuré' };
    }

    const { data, error } = await resend.emails.send({
      from: process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>',
      to: workerEmail,
      subject: siteName
        ? `Bienvenue sur le chantier ${siteName} - ChantiFlow`
        : 'Bienvenue dans l\'équipe ChantiFlow',
      html: `
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Bienvenue sur ChantiFlow</title>
          </head>
          <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
              <h1 style="color: white; margin: 0; font-size: 28px;">Bienvenue sur ChantiFlow</h1>
            </div>
            
            <div style="background: #ffffff; padding: 30px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 10px 10px;">
              <p style="font-size: 16px; margin-bottom: 20px;">
                Bonjour <strong>${workerName}</strong>,
              </p>
              
              ${siteName ? `
                <p style="font-size: 16px; margin-bottom: 20px;">
                  Vous avez été ajouté${managerName ? ` par ${managerName}` : ''} à l'équipe du chantier <strong>${siteName}</strong>.
                </p>
              ` : `
                <p style="font-size: 16px; margin-bottom: 20px;">
                  Vous avez été ajouté${managerName ? ` par ${managerName}` : ''} à l'équipe ChantiFlow.
                </p>
              `}
              
              <p style="font-size: 16px; margin-bottom: 20px;">
                Vous pouvez maintenant envoyer des rapports de terrain directement depuis votre téléphone.
              </p>
              
              <div style="background: #f9fafb; padding: 20px; border-radius: 8px; margin: 30px 0; border-left: 4px solid #667eea;">
                <h2 style="margin-top: 0; color: #1f2937; font-size: 20px;">Comment envoyer un rapport ?</h2>
                <ol style="margin: 10px 0; padding-left: 20px; color: #4b5563;">
                  <li style="margin-bottom: 10px;">Cliquez sur le lien ci-dessous ou scannez le QR code sur le chantier</li>
                  <li style="margin-bottom: 10px;">Sélectionnez la tâche concernée</li>
                  <li style="margin-bottom: 10px;">Ajoutez une photo et décrivez l'avancement</li>
                  <li style="margin-bottom: 10px;">Envoyez votre rapport au chef de chantier</li>
                </ol>
              </div>
              
              <div style="text-align: center; margin: 30px 0;">
                <a href="${reportUrl}" 
                   style="display: inline-block; background: #667eea; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: 600; font-size: 16px;">
                  Accéder au formulaire de rapport
                </a>
              </div>
              
              ${siteId ? `
                <p style="font-size: 14px; color: #6b7280; margin-top: 30px; text-align: center;">
                  Lien direct : <a href="${reportUrl}" style="color: #667eea;">${reportUrl}</a>
                </p>
              ` : ''}
              
              <p style="font-size: 14px; color: #6b7280; margin-top: 30px; border-top: 1px solid #e5e7eb; padding-top: 20px;">
                Si vous avez des questions, contactez votre chef de chantier.
              </p>
            </div>
            
            <div style="text-align: center; margin-top: 20px; color: #9ca3af; font-size: 12px;">
              <p>ChantiFlow - Gestion de chantier simplifiée</p>
            </div>
          </body>
        </html>
      `,
    });

    if (error) {
      console.error('Erreur envoi email:', error);
      return { success: false, error: error.message };
    }

    return { success: true, data };
  } catch (error) {
    console.error('Erreur envoi email:', error);
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
  if (!process.env.RESEND_API_KEY) {
    console.warn('RESEND_API_KEY non configuré, email non envoyé');
    return { success: false, error: 'Service email non configuré' };
  }

  try {
    // Vérifier que Resend est configuré
    if (!process.env.RESEND_API_KEY) {
      console.warn('RESEND_API_KEY non configuré, email non envoyé');
      return { success: false, error: 'Service email non configuré' };
    }

    const { data, error } = await resend.emails.send({
      from: process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>',
      to: managerEmail,
      subject: `Nouveau rapport - ${taskTitle} - ${siteName}`,
      html: `
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Nouveau rapport</title>
          </head>
          <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
              <h1 style="color: white; margin: 0; font-size: 28px;">Nouveau rapport reçu</h1>
            </div>
            
            <div style="background: #ffffff; padding: 30px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 10px 10px;">
              <p style="font-size: 16px; margin-bottom: 20px;">
                Bonjour${managerName ? ` ${managerName}` : ''},
              </p>
              
              <p style="font-size: 16px; margin-bottom: 20px;">
                <strong>${workerName}</strong>${workerEmail ? ` (${workerEmail})` : ''} a envoyé un nouveau rapport pour la tâche :
              </p>
              
              <div style="background: #f9fafb; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #10b981;">
                <p style="margin: 0; font-size: 18px; font-weight: 600; color: #1f2937;">
                  ${taskTitle}
                </p>
                <p style="margin: 10px 0 0 0; font-size: 14px; color: #6b7280;">
                  Chantier : ${siteName}
                </p>
              </div>
              
              <div style="text-align: center; margin: 30px 0;">
                <a href="${reportUrl}" 
                   style="display: inline-block; background: #10b981; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: 600; font-size: 16px;">
                  Voir le rapport
                </a>
              </div>
              
              <p style="font-size: 14px; color: #6b7280; margin-top: 30px; border-top: 1px solid #e5e7eb; padding-top: 20px;">
                Vous recevez cet email car vous êtes le gestionnaire du chantier ${siteName}.
              </p>
            </div>
            
            <div style="text-align: center; margin-top: 20px; color: #9ca3af; font-size: 12px;">
              <p>ChantiFlow - Gestion de chantier simplifiée</p>
            </div>
          </body>
        </html>
      `,
    });

    if (error) {
      console.error('Erreur envoi email notification:', error);
      return { success: false, error: error.message };
    }

    return { success: true, data };
  } catch (error) {
    console.error('Erreur envoi email notification:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Erreur inconnue' };
  }
}

