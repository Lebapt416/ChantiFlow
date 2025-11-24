/**
 * Templates d'emails professionnels pour ChantiFlow
 * Design moderne et cohérent avec le branding
 */

const PRIMARY_COLOR = '#10b981'; // Emerald 500
const PRIMARY_DARK = '#059669'; // Emerald 600
const SECONDARY_COLOR = '#3f3f46'; // Zinc 700
const TEXT_PRIMARY = '#18181b'; // Zinc 900
const TEXT_SECONDARY = '#71717a'; // Zinc 500
const BG_LIGHT = '#f4f4f5'; // Zinc 100
const BORDER_COLOR = '#e4e4e7'; // Zinc 200

/**
 * Template de base pour tous les emails
 */
function getEmailBase(headerTitle: string, headerColor: string = PRIMARY_COLOR) {
  return `
    <!DOCTYPE html>
    <html lang="fr">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <title>${headerTitle} - ChantiFlow</title>
      </head>
      <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #fafafa; line-height: 1.6;">
        <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #fafafa; padding: 40px 20px;">
          <tr>
            <td align="center">
              <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="600" style="max-width: 600px; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);">
                
                <!-- Header -->
                <tr>
                  <td style="background: ${headerColor}; padding: 40px 30px; text-align: center;">
                    <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700; letter-spacing: -0.5px;">
                      ${headerTitle}
                    </h1>
                  </td>
                </tr>
                
                <!-- Branding Bar -->
                <tr>
                  <td style="background: linear-gradient(135deg, ${PRIMARY_COLOR} 0%, ${PRIMARY_DARK} 100%); padding: 12px 30px; text-align: right;">
                    <p style="margin: 0; color: #ffffff; font-size: 16px; font-weight: 700; letter-spacing: 0.5px;">
                      ChantiFlow
                    </p>
                    <p style="margin: 4px 0 0 0; color: rgba(255, 255, 255, 0.9); font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">
                      Gestion intelligente BTP
                    </p>
                  </td>
                </tr>
                
                <!-- Content -->
                <tr>
                  <td style="padding: 40px 30px;">
`;
}

function getEmailFooter() {
  return `
                  </td>
                </tr>
                
                <!-- Footer -->
                <tr>
                  <td style="background-color: ${BG_LIGHT}; padding: 30px; text-align: center; border-top: 1px solid ${BORDER_COLOR};">
                    <p style="margin: 0 0 8px 0; color: ${TEXT_SECONDARY}; font-size: 13px; font-weight: 600;">
                      ChantiFlow
                    </p>
                    <p style="margin: 0; color: ${TEXT_SECONDARY}; font-size: 12px;">
                      Gestion de chantier simplifiée avec intelligence artificielle
                    </p>
                    <p style="margin: 16px 0 0 0; color: ${TEXT_SECONDARY}; font-size: 11px;">
                      © ${new Date().getFullYear()} ChantiFlow. Tous droits réservés.
                    </p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
        </table>
      </body>
    </html>
  `;
}

/**
 * Template pour email de bienvenue worker
 */
export function getWorkerWelcomeEmailTemplate({
  workerName,
  siteName,
  managerName,
  accessCode,
  reportUrl,
  siteId,
}: {
  workerName: string;
  siteName?: string;
  managerName?: string;
  accessCode?: string;
  reportUrl: string;
  siteId?: string;
}) {
  const headerTitle = siteName ? `Bienvenue sur le chantier ${siteName}` : 'Bienvenue dans l\'équipe ChantiFlow';
  
  let content = getEmailBase(headerTitle);
  
  content += `
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Bonjour <strong style="color: ${PRIMARY_COLOR};">${workerName}</strong>,
                    </p>
  `;

  if (siteName) {
    content += `
                    <p style="margin: 0 0 20px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Vous avez été assigné${managerName ? ` par <strong>${managerName}</strong>` : ''} au chantier <strong style="color: ${PRIMARY_COLOR};">${siteName}</strong>.
                    </p>
    `;
    
    if (accessCode && accessCode.trim() !== '') {
      content += `
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Vous pouvez maintenant accéder aux informations de ce chantier et envoyer des rapports en utilisant votre code d'accès unique ci-dessous.
                    </p>
                    
                    <!-- Code d'accès -->
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="margin: 32px 0; background: linear-gradient(135deg, ${PRIMARY_COLOR} 0%, ${PRIMARY_DARK} 100%); border-radius: 12px; padding: 32px 24px; box-shadow: 0 8px 16px rgba(16, 185, 129, 0.2);">
                      <tr>
                        <td align="center" style="padding: 0;">
                          <p style="margin: 0 0 12px 0; color: #ffffff; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px;">
                            Votre code d'accès unique
                          </p>
                          <p style="margin: 16px 0; color: #ffffff; font-size: 42px; font-weight: 700; letter-spacing: 8px; font-family: 'Courier New', monospace; text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);">
                            ${String(accessCode).toUpperCase()}
                          </p>
                          <p style="margin: 12px 0 0 0; color: rgba(255, 255, 255, 0.95); font-size: 13px; font-weight: 500;">
                            ⚠️ Conservez ce code précieusement
                          </p>
                          <p style="margin: 6px 0 0 0; color: rgba(255, 255, 255, 0.85); font-size: 12px;">
                            Vous devrez entrer ce code après avoir scanné le QR code du chantier
                          </p>
                        </td>
                      </tr>
                    </table>
      `;
    } else if (siteId) {
      content += `
                    <div style="background: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 20px; margin: 24px 0; text-align: center;">
                      <p style="margin: 0; color: #92400e; font-size: 14px; font-weight: 600;">ℹ️ Code d'accès à venir</p>
                      <p style="margin: 8px 0 0 0; color: #92400e; font-size: 12px;">Vous recevrez votre code d'accès par email lorsque vous serez assigné à un chantier spécifique.</p>
                    </div>
      `;
    }
  } else {
    content += `
                    <p style="margin: 0 0 20px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Vous avez été ajouté${managerName ? ` par <strong>${managerName}</strong>` : ''} à l'équipe ChantiFlow.
                    </p>
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Lorsque vous serez assigné à un chantier spécifique, vous recevrez un code d'accès unique pour accéder aux informations de ce chantier.
                    </p>
    `;
  }

  content += `
                    <!-- Instructions -->
                    <div style="background: ${BG_LIGHT}; border-left: 4px solid ${PRIMARY_COLOR}; border-radius: 8px; padding: 24px; margin: 32px 0;">
                      <h2 style="margin: 0 0 16px 0; color: ${TEXT_PRIMARY}; font-size: 18px; font-weight: 700;">
                        📋 Comment envoyer un rapport ?
                      </h2>
                      <ol style="margin: 0; padding-left: 20px; color: ${TEXT_SECONDARY}; font-size: 15px; line-height: 1.8;">
  `;

  if (siteId) {
    content += `
                        <li style="margin-bottom: 8px;">Scannez le QR code sur le chantier</li>
                        ${accessCode ? '<li style="margin-bottom: 8px;">Entrez votre code d\'accès unique (ci-dessus)</li>' : '<li style="margin-bottom: 8px;">Vous recevrez votre code d\'accès par email</li>'}
                        <li style="margin-bottom: 8px;">Sélectionnez la tâche concernée</li>
                        <li style="margin-bottom: 8px;">Ajoutez une photo et décrivez l'avancement</li>
                        <li style="margin-bottom: 8px;">Envoyez votre rapport au chef de chantier</li>
    `;
  } else {
    content += `
                        <li style="margin-bottom: 8px;">Vous avez été ajouté à l'équipe ChantiFlow</li>
                        <li style="margin-bottom: 8px;">Lorsque vous serez assigné à un chantier, vous recevrez un code d'accès unique</li>
                        <li style="margin-bottom: 8px;">Avec ce code, vous pourrez scanner le QR code du chantier et envoyer des rapports</li>
    `;
  }

  content += `
                      </ol>
                    </div>
                    
                    <!-- CTA Button -->
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="margin: 32px 0;">
                      <tr>
                        <td align="center" style="padding: 0;">
                          <a href="${reportUrl}" 
                             style="display: inline-block; background: ${PRIMARY_COLOR}; color: #ffffff; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3); transition: all 0.2s;">
                            Accéder au formulaire de rapport
                          </a>
                        </td>
                      </tr>
                    </table>
  `;

  if (siteId) {
    content += `
                    <p style="margin: 24px 0 0 0; color: ${TEXT_SECONDARY}; font-size: 13px; text-align: center;">
                      Lien direct : <a href="${reportUrl}" style="color: ${PRIMARY_COLOR}; text-decoration: none; font-weight: 500;">${reportUrl}</a>
                    </p>
    `;
  }

  content += `
                    <p style="margin: 32px 0 0 0; color: ${TEXT_SECONDARY}; font-size: 14px; border-top: 1px solid ${BORDER_COLOR}; padding-top: 24px;">
                      Si vous avez des questions, contactez votre chef de chantier.
                    </p>
  `;

  content += getEmailFooter();
  return content;
}

/**
 * Template pour notification de nouveau rapport
 */
export function getReportNotificationEmailTemplate({
  managerName,
  workerName,
  workerEmail,
  taskTitle,
  siteName,
  reportUrl,
}: {
  managerName?: string;
  workerName: string;
  workerEmail?: string;
  taskTitle: string;
  siteName: string;
  reportUrl: string;
}) {
  let content = getEmailBase('Nouveau rapport reçu', PRIMARY_COLOR);
  
  content += `
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Bonjour${managerName ? ` <strong>${managerName}</strong>` : ''},
                    </p>
                    
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      <strong>${workerName}</strong>${workerEmail ? ` (${workerEmail})` : ''} a envoyé un nouveau rapport pour la tâche :
                    </p>
                    
                    <!-- Info Box -->
                    <div style="background: ${BG_LIGHT}; border-left: 4px solid ${PRIMARY_COLOR}; border-radius: 8px; padding: 24px; margin: 24px 0;">
                      <p style="margin: 0 0 8px 0; color: ${TEXT_PRIMARY}; font-size: 18px; font-weight: 700;">
                        ${taskTitle}
                      </p>
                      <p style="margin: 0; color: ${TEXT_SECONDARY}; font-size: 14px;">
                        Chantier : <strong>${siteName}</strong>
                      </p>
                    </div>
                    
                    <!-- CTA Button -->
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="margin: 32px 0;">
                      <tr>
                        <td align="center" style="padding: 0;">
                          <a href="${reportUrl}" 
                             style="display: inline-block; background: ${PRIMARY_COLOR}; color: #ffffff; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);">
                            Voir le rapport
                          </a>
                        </td>
                      </tr>
                    </table>
                    
                    <p style="margin: 32px 0 0 0; color: ${TEXT_SECONDARY}; font-size: 14px; border-top: 1px solid ${BORDER_COLOR}; padding-top: 24px;">
                      Vous recevez cet email car vous êtes le gestionnaire du chantier <strong>${siteName}</strong>.
                    </p>
  `;

  content += getEmailFooter();
  return content;
}

/**
 * Template pour compte créé
 */
export function getAccountCreatedEmailTemplate({
  userEmail,
  temporaryPassword,
  loginUrl,
}: {
  userEmail: string;
  temporaryPassword: string;
  loginUrl: string;
}) {
  let content = getEmailBase('Bienvenue sur ChantiFlow', PRIMARY_COLOR);
  
  content += `
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Bonjour,
                    </p>
                    
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Votre paiement a été effectué avec succès et votre compte ChantiFlow a été créé automatiquement !
                    </p>
                    
                    <!-- Credentials Box -->
                    <div style="background: #fef3c7; border: 2px solid #f59e0b; border-radius: 12px; padding: 24px; margin: 32px 0;">
                      <p style="margin: 0 0 16px 0; color: #92400e; font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
                        🔐 Vos identifiants de connexion
                      </p>
                      <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                        <tr>
                          <td style="padding: 0 0 12px 0;">
                            <p style="margin: 0; color: #92400e; font-size: 14px;"><strong>Email :</strong> ${userEmail}</p>
                          </td>
                        </tr>
                        <tr>
                          <td style="padding: 0;">
                            <p style="margin: 0 0 8px 0; color: #92400e; font-size: 14px;"><strong>Mot de passe temporaire :</strong></p>
                            <div style="background: #ffffff; border: 1px solid #f59e0b; border-radius: 8px; padding: 16px; text-align: center; margin-top: 8px;">
                              <p style="margin: 0; color: #92400e; font-size: 20px; font-weight: 700; font-family: 'Courier New', monospace; letter-spacing: 3px;">
                                ${temporaryPassword}
                              </p>
                            </div>
                            <p style="margin: 12px 0 0 0; color: #92400e; font-size: 12px; font-style: italic; text-align: center;">
                              ⚠️ Pour des raisons de sécurité, nous vous recommandons de changer ce mot de passe après votre première connexion.
                            </p>
                          </td>
                        </tr>
                      </table>
                    </div>
                    
                    <p style="margin: 0 0 32px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Votre abonnement est déjà activé et vous avez accès à toutes les fonctionnalités de votre plan.
                    </p>
                    
                    <!-- CTA Button -->
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="margin: 32px 0;">
                      <tr>
                        <td align="center" style="padding: 0;">
                          <a href="${loginUrl}" 
                             style="display: inline-block; background: ${PRIMARY_COLOR}; color: #ffffff; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);">
                            Se connecter maintenant
                          </a>
                        </td>
                      </tr>
                    </table>
                    
                    <p style="margin: 32px 0 0 0; color: ${TEXT_SECONDARY}; font-size: 14px; border-top: 1px solid ${BORDER_COLOR}; padding-top: 24px;">
                      Si vous avez des questions, n'hésitez pas à nous contacter à <a href="mailto:contact@chantiflow.com" style="color: ${PRIMARY_COLOR}; text-decoration: none; font-weight: 500;">contact@chantiflow.com</a>.
                    </p>
  `;

  content += getEmailFooter();
  return content;
}

/**
 * Template pour chantier terminé
 */
export function getSiteCompletedEmailTemplate({
  workerName,
  siteName,
}: {
  workerName: string;
  siteName: string;
}) {
  let content = getEmailBase('Chantier terminé', PRIMARY_COLOR);
  
  content += `
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Bonjour <strong>${workerName}</strong>,
                    </p>
                    
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Nous vous informons que le chantier <strong style="color: ${PRIMARY_COLOR};">${siteName}</strong> a été marqué comme terminé.
                    </p>
                    
                    <!-- Success Box -->
                    <div style="background: #f0fdf4; border: 2px solid ${PRIMARY_COLOR}; border-radius: 12px; padding: 32px; margin: 32px 0; text-align: center;">
                      <p style="margin: 0 0 12px 0; color: #065f46; font-size: 18px; font-weight: 700;">
                        ✅ Le chantier est maintenant terminé
                      </p>
                      <p style="margin: 0; color: #047857; font-size: 15px; line-height: 1.6;">
                        Vous avez été retiré de ce chantier. Merci pour votre travail !
                      </p>
                    </div>
                    
                    <p style="margin: 32px 0 0 0; color: ${TEXT_SECONDARY}; font-size: 14px; border-top: 1px solid ${BORDER_COLOR}; padding-top: 24px;">
                      Merci pour votre contribution au projet <strong>${siteName}</strong>.
                    </p>
                    
                    <p style="margin: 24px 0 0 0; color: ${TEXT_SECONDARY}; font-size: 14px;">
                      Si vous avez des questions ou besoin d'informations supplémentaires, n'hésitez pas à contacter votre chef de chantier.
                    </p>
  `;

  content += getEmailFooter();
  return content;
}

/**
 * Template pour demande d'ajout à l'équipe (en attente)
 */
export function getTeamJoinConfirmationEmailTemplate({
  workerName,
  managerName,
}: {
  workerName: string;
  managerName?: string;
}) {
  let content = getEmailBase('Demande reçue !', PRIMARY_COLOR);
  
  content += `
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Bonjour <strong>${workerName}</strong>,
                    </p>
                    
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Votre demande d'ajout à l'équipe a bien été reçue. Nous vous remercions de votre intérêt !
                    </p>
                    
                    <!-- Pending Box -->
                    <div style="background: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 8px; padding: 24px; margin: 32px 0;">
                      <p style="margin: 0 0 12px 0; color: #92400e; font-size: 16px; font-weight: 700;">
                        ⏳ En attente de validation
                      </p>
                      <p style="margin: 0; color: #78350f; font-size: 15px; line-height: 1.6;">
                        Votre demande est actuellement en attente de validation par le chef de chantier${managerName ? ` <strong>${managerName}</strong>` : ''}. Vous recevrez un email de confirmation une fois votre demande approuvée.
                      </p>
                    </div>
                    
                    <p style="margin: 32px 0 0 0; color: ${TEXT_SECONDARY}; font-size: 14px; border-top: 1px solid ${BORDER_COLOR}; padding-top: 24px;">
                      En attendant, vous pouvez préparer vos informations et documents nécessaires pour votre intégration à l'équipe.
                    </p>
  `;

  content += getEmailFooter();
  return content;
}

/**
 * Template pour approbation d'équipe
 */
export function getTeamApprovalEmailTemplate({
  workerName,
  managerName,
}: {
  workerName: string;
  managerName?: string;
}) {
  let content = getEmailBase('Demande approuvée !', PRIMARY_COLOR);
  
  content += `
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Bonjour <strong>${workerName}</strong>,
                    </p>
                    
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Excellente nouvelle ! Votre demande d'ajout à l'équipe a été approuvée${managerName ? ` par <strong>${managerName}</strong>` : ''}.
                    </p>
                    
                    <!-- Success Box -->
                    <div style="background: #f0fdf4; border: 2px solid ${PRIMARY_COLOR}; border-radius: 12px; padding: 32px; margin: 32px 0; text-align: center;">
                      <p style="margin: 0 0 12px 0; color: #065f46; font-size: 18px; font-weight: 700;">
                        ✅ Vous faites maintenant partie de l'équipe !
                      </p>
                      <p style="margin: 0; color: #047857; font-size: 15px; line-height: 1.6;">
                        Vous pouvez maintenant être assigné à des chantiers et commencer à travailler.
                      </p>
                    </div>
                    
                    <p style="margin: 32px 0 0 0; color: ${TEXT_SECONDARY}; font-size: 14px; border-top: 1px solid ${BORDER_COLOR}; padding-top: 24px;">
                      Vous recevrez un email avec votre code d'accès unique lorsque vous serez assigné à un chantier spécifique.
                    </p>
  `;

  content += getEmailFooter();
  return content;
}

/**
 * Template pour refus d'équipe
 */
export function getTeamRejectionEmailTemplate({
  workerName,
  managerName,
}: {
  workerName: string;
  managerName?: string;
}) {
  let content = getEmailBase('Demande refusée', '#ef4444');
  
  content += `
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Bonjour <strong>${workerName}</strong>,
                    </p>
                    
                    <p style="margin: 0 0 24px 0; color: ${TEXT_PRIMARY}; font-size: 16px; line-height: 1.6;">
                      Nous vous informons que votre demande d'ajout à l'équipe a été refusée${managerName ? ` par <strong>${managerName}</strong>` : ''}.
                    </p>
                    
                    <!-- Rejection Box -->
                    <div style="background: #fef2f2; border: 2px solid #ef4444; border-radius: 12px; padding: 32px; margin: 32px 0; text-align: center;">
                      <p style="margin: 0 0 12px 0; color: #991b1b; font-size: 18px; font-weight: 700;">
                        ❌ Votre demande n'a pas été retenue
                      </p>
                      <p style="margin: 0; color: #b91c1c; font-size: 15px; line-height: 1.6;">
                        Si vous avez des questions, n'hésitez pas à contacter directement le chef de chantier.
                      </p>
                    </div>
                    
                    <p style="margin: 32px 0 0 0; color: ${TEXT_SECONDARY}; font-size: 14px; border-top: 1px solid ${BORDER_COLOR}; padding-top: 24px;">
                      Nous vous remercions de votre intérêt pour rejoindre l'équipe.
                    </p>
  `;

  content += getEmailFooter();
  return content;
}

