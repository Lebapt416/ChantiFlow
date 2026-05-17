import { NextRequest, NextResponse } from 'next/server';
import { Resend } from 'resend';
import { randomUUID } from 'crypto';

const resend = process.env.RESEND_API_KEY ? new Resend(process.env.RESEND_API_KEY) : null;
const CONTACT_EMAIL = 'chantiflowct@gmail.com';

export async function POST(request: NextRequest) {
  try {
    // Vérifier que l'utilisateur est admin (sinon = open mail relay)
    const { createSupabaseServerClient } = await import('@/lib/supabase/server');
    const { isAdmin } = await import('@/lib/admin');
    const supabase = await createSupabaseServerClient();
    const { data: { user } } = await supabase.auth.getUser();

    if (!user || !isAdmin(user.email)) {
      return NextResponse.json(
        { error: 'Non autorisé.' },
        { status: 403 }
      );
    }

    const body = await request.json();
    const { to, subject, message, originalMessage } = body;

    // Validation
    if (!to || !message) {
      return NextResponse.json(
        { error: 'Destinataire et message sont requis.' },
        { status: 400 }
      );
    }

    // Validation email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(to)) {
      return NextResponse.json(
        { error: 'Format d\'email invalide.' },
        { status: 400 }
      );
    }

    if (!resend) {
      console.error('RESEND_API_KEY non configuré');
      return NextResponse.json(
        { error: 'Service d\'envoi d\'email non configuré.' },
        { status: 500 }
      );
    }

    // Envoyer la réponse
    const { error } = await resend.emails.send({
      from: `ChantiFlow <${CONTACT_EMAIL}>`,
      to: to,
      replyTo: CONTACT_EMAIL,
      subject: subject || 'Re: Votre message de contact',
      headers: {
        'X-Entity-Ref-ID': randomUUID(),
        'X-Priority': '1',
        'X-MSMail-Priority': 'High',
      },
      tags: [
        { name: 'category', value: 'contact-reply' },
        { name: 'source', value: 'chantiflow' },
      ],
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
              .message-box {
                background: #f9fafb;
                border-left: 4px solid #10b981;
                padding: 16px;
                margin-top: 8px;
                border-radius: 4px;
              }
              .original-message {
                background: #f3f4f6;
                border-left: 4px solid #9ca3af;
                padding: 16px;
                margin-top: 16px;
                border-radius: 4px;
                font-size: 14px;
                color: #6b7280;
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
              <h1 style="margin: 0; font-size: 24px;">Réponse ChantiFlow</h1>
            </div>
            <div class="content">
              <div class="message-box">
                ${escapeHtml(message).replace(/\n/g, '<br>')}
              </div>
              ${originalMessage ? `
              <div class="original-message">
                <strong>Message original :</strong><br>
                ${escapeHtml(originalMessage).replace(/\n/g, '<br>')}
              </div>
              ` : ''}
            </div>
            <div class="footer">
              <p>Cet email a été envoyé depuis ChantiFlow.</p>
              <p>Pour toute question, contactez-nous à ${CONTACT_EMAIL}</p>
            </div>
          </body>
        </html>
      `,
      text: `
${message}

${originalMessage ? `\n---\nMessage original :\n${originalMessage}` : ''}

---
Cet email a été envoyé depuis ChantiFlow.
Pour toute question, contactez-nous à ${CONTACT_EMAIL}
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
      { success: true, message: 'Réponse envoyée avec succès' },
      { status: 200 }
    );
  } catch (error) {
    console.error('Erreur API contact reply:', error);
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

