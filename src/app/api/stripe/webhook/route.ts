import { NextResponse } from 'next/server';
import { headers } from 'next/headers';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';
import { stripe } from '@/lib/stripe';

export async function POST(request: Request) {
  if (!stripe) {
    return NextResponse.json(
      { error: 'Stripe non configuré' },
      { status: 500 }
    );
  }

  const body = await request.text();
  const headersList = await headers();
  const signature = headersList.get('stripe-signature');

  if (!signature) {
    return NextResponse.json(
      { error: 'Signature manquante' },
      { status: 400 }
    );
  }

  let event;

  try {
    event = stripe.webhooks.constructEvent(
      body,
      signature,
      process.env.STRIPE_WEBHOOK_SECRET || ''
    );
  } catch (err) {
    console.error('Erreur vérification webhook Stripe:', err);
    return NextResponse.json(
      { error: 'Signature invalide' },
      { status: 400 }
    );
  }

  const admin = createSupabaseAdminClient();

  // Gérer les événements Stripe
  switch (event.type) {
    case 'checkout.session.completed': {
      const session = event.data.object as any;
      const userId = session.metadata?.userId;
      const plan = session.metadata?.plan;

      if (userId && plan) {
        // Mettre à jour le plan de l'utilisateur
        const { error } = await admin.auth.admin.updateUserById(userId, {
          user_metadata: {
            plan,
            plan_updated_at: new Date().toISOString(),
            stripe_customer_id: session.customer,
            stripe_subscription_id: session.subscription,
          },
        });

        if (error) {
          console.error('Erreur mise à jour plan:', error);
        }
      }
      break;
    }

    case 'customer.subscription.updated':
    case 'customer.subscription.deleted': {
      const subscription = event.data.object as any;
      const customerId = subscription.customer;

      // Trouver l'utilisateur par customer_id
      // Note: Cette recherche nécessiterait une table de mapping ou une recherche dans les métadonnées
      // Pour simplifier, on peut stocker le customer_id dans user_metadata
      break;
    }

    default:
      console.log(`Événement non géré: ${event.type}`);
  }

  return NextResponse.json({ received: true });
}

