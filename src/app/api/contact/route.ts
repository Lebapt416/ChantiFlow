import { NextRequest, NextResponse } from 'next/server';
import { Resend } from 'resend';

const resend = process.env.RESEND_API_KEY ? new Resend(process.env.RESEND_API_KEY) : null;
const CONTACT_EMAIL = process.env.CONTACT_EMAIL || 'contact@chantiflow.com';
const RESEND_FROM_EMAIL = process.env.RESEND_FROM_EMAIL || 'ChantiFlow <onboarding@resend.dev>';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { name, email, company, message } = body;

    // Validation
    if (!name || !email || !message) {
      return NextResponse.json(
        { error: 'Tous les champs obligatoires doivent être remplis.' },
        { status: 400 }
      );
    }

    // Validation email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { error: 'Format d\'email invalide.' },
        { status: 400 }
      );
    }

    if (!resend) {
      console.error('RESEND_API_KEY non configuré');
      return NextResponse.json(
        { error: 'Service d\'envoi d\'email non configuré. Veuillez réessayer plus tard.' },
        { status: 500 }
      );
    }

    // Envoyer l'email
    const { data, error } = await resend.emails.send({
      from: RESEND_FROM_EMAIL,
      to: CONTACT_EMAIL,
      replyTo: email,
      subject: `Nouveau message de contact${company ? ` - ${company}` : ''}`,
      html: `
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8">
            <style>
              body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #1f2937;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
              }
              .header {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 24px;
                border-radius: 8px 8px 0 0;
                text-align: center;
              }
              .content {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-top: none;
                padding: 24px;
                border-radius: 0 0 8px 8px;
              }
              .field {
                margin-bottom: 16px;
              }
              .label {
                font-weight: 600;
                color: #374151;
                font-size: 14px;
                margin-bottom: 4px;
                display: block;
              }
              .value {
                color: #1f2937;
                font-size: 16px;
              }
              .message-box {
                background: #f9fafb;
                border-left: 4px solid #10b981;
                padding: 16px;
                margin-top: 8px;
                border-radius: 4px;
              }
              .footer {
                margin-top: 24px;
                padding-top: 24px;
                border-top: 1px solid #e5e7eb;
                font-size: 12px;
                color: #6b7280;
                text-align: center;
              }
            </style>
          </head>
          <body>
            <div class="header">
              <h1 style="margin: 0; font-size: 24px;">Nouveau message de contact</h1>
            </div>
            <div class="content">
              <div class="field">
                <span class="label">Nom :</span>
                <span class="value">${escapeHtml(name)}</span>
              </div>
              
              <div class="field">
                <span class="label">Email :</span>
                <span class="value"><a href="mailto:${escapeHtml(email)}">${escapeHtml(email)}</a></span>
              </div>
              
              ${company ? `
              <div class="field">
                <span class="label">Entreprise :</span>
                <span class="value">${escapeHtml(company)}</span>
              </div>
              ` : ''}
              
              <div class="field">
                <span class="label">Message :</span>
                <div class="message-box">
                  ${escapeHtml(message).replace(/\n/g, '<br>')}
                </div>
              </div>
            </div>
            <div class="footer">
              <p>Ce message a été envoyé depuis le formulaire de contact de ChantiFlow.</p>
              <p>Vous pouvez répondre directement à cet email pour contacter ${escapeHtml(name)}.</p>
            </div>
          </body>
        </html>
      `,
      text: `
Nouveau message de contact${company ? ` - ${company}` : ''}

Nom : ${name}
Email : ${email}
${company ? `Entreprise : ${company}\n` : ''}
Message :
${message}

---
Ce message a été envoyé depuis le formulaire de contact de ChantiFlow.
Vous pouvez répondre directement à cet email pour contacter ${name}.
      `,
    });

    if (error) {
      console.error('Erreur Resend:', error);
      return NextResponse.json(
        { error: 'Erreur lors de l\'envoi de l\'email. Veuillez réessayer.' },
        { status: 500 }
      );
    }

    return NextResponse.json(
      { success: true, message: 'Message envoyé avec succès' },
      { status: 200 }
    );
  } catch (error) {
    console.error('Erreur API contact:', error);
    return NextResponse.json(
      { error: 'Une erreur est survenue. Veuillez réessayer.' },
      { status: 500 }
    );
  }
}

function escapeHtml(text: string): string {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  };
  return text.replace(/[&<>"']/g, (m) => map[m]);
}

