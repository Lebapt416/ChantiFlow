import { Resend } from 'resend';

// Initialiser Resend seulement si la clé API est disponible
const resend = process.env.RESEND_API_KEY ? new Resend(process.env.RESEND_API_KEY) : null;

export async function sendWorkerWelcomeEmail({
  workerEmail,
  workerName,
  siteName,
  siteId,
  managerName,
  accessCode,
}: {
  workerEmail: string;
  workerName: string;
  siteName?: string;
  siteId?: string;
  managerName?: string;
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
                  Vous avez été assigné${managerName ? ` par ${managerName}` : ''} au chantier <strong>${siteName}</strong>.
                </p>
                ${accessCode ? `
                  <p style="font-size: 16px; margin-bottom: 20px;">
                    Vous pouvez maintenant accéder aux informations de ce chantier et envoyer des rapports en utilisant votre code d'accès unique ci-dessous.
                  </p>
                ` : ''}
              ` : `
                <p style="font-size: 16px; margin-bottom: 20px;">
                  Vous avez été ajouté${managerName ? ` par ${managerName}` : ''} à l'équipe ChantiFlow.
                </p>
                <p style="font-size: 16px; margin-bottom: 20px;">
                  Lorsque vous serez assigné à un chantier spécifique, vous recevrez un code d'accès unique pour accéder aux informations de ce chantier.
                </p>
              `}
              
              ${accessCode && accessCode.trim() !== '' ? `
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 25px; border-radius: 8px; margin: 30px 0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                  <p style="color: white; font-size: 14px; margin: 0 0 10px 0; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Votre code d'accès unique pour ce chantier</p>
                  <p style="color: white; font-size: 36px; margin: 15px 0; font-weight: bold; letter-spacing: 6px; font-family: 'Courier New', monospace; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${String(accessCode).toUpperCase()}</p>
                  <p style="color: rgba(255,255,255,0.95); font-size: 13px; margin: 15px 0 0 0; font-weight: 500;">⚠️ Conservez ce code précieusement</p>
                  <p style="color: rgba(255,255,255,0.9); font-size: 12px; margin: 5px 0 0 0;">Vous devrez entrer ce code après avoir scanné le QR code du chantier</p>
                </div>
              ` : siteId ? `
                <div style="background: #fef3c7; border: 2px solid #f59e0b; padding: 15px; border-radius: 8px; margin: 30px 0; text-align: center;">
                  <p style="color: #92400e; font-size: 14px; margin: 0; font-weight: 600;">ℹ️ Code d'accès à venir</p>
                  <p style="color: #92400e; font-size: 12px; margin: 5px 0 0 0;">Vous recevrez votre code d'accès par email lorsque vous serez assigné à un chantier spécifique.</p>
                </div>
              ` : ''}
              
              ${!accessCode ? `
                <p style="font-size: 16px; margin-bottom: 20px;">
                  Vous pouvez maintenant envoyer des rapports de terrain directement depuis votre téléphone.
                </p>
              ` : ''}
              
              <div style="background: #f9fafb; padding: 20px; border-radius: 8px; margin: 30px 0; border-left: 4px solid #667eea;">
                <h2 style="margin-top: 0; color: #1f2937; font-size: 20px;">Comment envoyer un rapport ?</h2>
                <ol style="margin: 10px 0; padding-left: 20px; color: #4b5563;">
                  ${siteId ? `
                    <li style="margin-bottom: 10px;">Scannez le QR code sur le chantier</li>
                    ${accessCode ? '<li style="margin-bottom: 10px;">Entrez votre code d\'accès unique (ci-dessus)</li>' : '<li style="margin-bottom: 10px;">Vous recevrez votre code d\'accès par email</li>'}
                    <li style="margin-bottom: 10px;">Sélectionnez la tâche concernée</li>
                    <li style="margin-bottom: 10px;">Ajoutez une photo et décrivez l'avancement</li>
                    <li style="margin-bottom: 10px;">Envoyez votre rapport au chef de chantier</li>
                  ` : `
                    <li style="margin-bottom: 10px;">Vous avez été ajouté à l'équipe ChantiFlow</li>
                    <li style="margin-bottom: 10px;">Lorsque vous serez assigné à un chantier, vous recevrez un code d'accès unique</li>
                    <li style="margin-bottom: 10px;">Avec ce code, vous pourrez scanner le QR code du chantier et envoyer des rapports</li>
                  `}
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
    const { data, error } = await resend.emails.send({
      from: process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>',
      to: userEmail,
      subject: 'Bienvenue sur ChantiFlow - Votre compte a été créé',
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
                Bonjour,
              </p>
              
              <p style="font-size: 16px; margin-bottom: 20px;">
                Votre paiement a été effectué avec succès et votre compte ChantiFlow a été créé automatiquement !
              </p>
              
              <div style="background: #fef3c7; border: 2px solid #f59e0b; padding: 20px; border-radius: 8px; margin: 30px 0;">
                <p style="color: #92400e; font-size: 14px; margin: 0 0 10px 0; font-weight: 600;">🔐 Vos identifiants de connexion</p>
                <p style="color: #92400e; font-size: 14px; margin: 0 0 5px 0;"><strong>Email :</strong> ${userEmail}</p>
                <p style="color: #92400e; font-size: 14px; margin: 0;"><strong>Mot de passe temporaire :</strong></p>
                <div style="background: #fff; border: 1px solid #f59e0b; padding: 15px; border-radius: 6px; margin-top: 10px; text-align: center;">
                  <p style="color: #92400e; font-size: 18px; margin: 0; font-weight: bold; font-family: 'Courier New', monospace; letter-spacing: 2px;">${temporaryPassword}</p>
                </div>
                <p style="color: #92400e; font-size: 12px; margin: 10px 0 0 0; font-style: italic;">
                  ⚠️ Pour des raisons de sécurité, nous vous recommandons de changer ce mot de passe après votre première connexion.
                </p>
              </div>
              
              <p style="font-size: 16px; margin-bottom: 20px;">
                Votre abonnement est déjà activé et vous avez accès à toutes les fonctionnalités de votre plan.
              </p>
              
              <div style="text-align: center; margin: 30px 0;">
                <a href="${loginUrl}" 
                   style="display: inline-block; background: #667eea; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: 600; font-size: 16px;">
                  Se connecter maintenant
                </a>
              </div>
              
              <p style="font-size: 14px; color: #6b7280; margin-top: 30px; border-top: 1px solid #e5e7eb; padding-top: 20px;">
                Si vous avez des questions, n'hésitez pas à nous contacter à <a href="mailto:contact@chantiflow.com" style="color: #667eea;">contact@chantiflow.com</a>.
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
      console.error('❌ Erreur envoi email compte créé:', error);
      return { success: false, error: error.message };
    }

    console.log('✅ Email identifiants envoyé à', userEmail);
    return { success: true, data };
  } catch (error: any) {
    console.error('❌ Exception envoi email compte créé:', error);
    return { success: false, error: error?.message || 'Erreur inconnue' };
  }
}

