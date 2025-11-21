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
      const customerEmail = session.customer_email || session.customer_details?.email;
      
      if (!customerEmail) {
        console.error('Email customer manquant dans la session');
        break;
      }

      // Déterminer le plan en fonction du prix acheté
      let plan: 'plus' | 'pro' | null = null;
      
      if (session.line_items?.data) {
        // Si on a les line_items, on peut déterminer le plan
        const priceId = session.line_items.data[0]?.price?.id;
        // Vous devrez mapper vos price IDs ici si nécessaire
        // Pour l'instant, on va utiliser une autre méthode
      }

      // Alternative : utiliser l'URL de la session ou les métadonnées du produit
      // Pour les liens buy.stripe.com, on peut vérifier le customer_email
      // et chercher l'utilisateur correspondant dans Supabase
      
      // Chercher l'utilisateur par email
      const { data: users } = await admin.auth.admin.listUsers();
      const user = users.users.find((u) => u.email === customerEmail);

      if (!user) {
        console.warn(`⚠️ Utilisateur non trouvé pour l'email: ${customerEmail}`);
        console.warn(`⚠️ Le paiement a été effectué mais aucun compte n'existe avec cet email.`);
        console.warn(`⚠️ L'utilisateur devra créer un compte avec cet email pour que le plan soit activé.`);
        // On pourrait créer un compte automatiquement ici, mais pour l'instant on log juste l'avertissement
        break;
      }

      // Déterminer le plan en fonction du montant payé
      // Pour les liens buy.stripe.com, on utilise le montant total
      const amountTotal = session.amount_total; // en centimes
      
      if (amountTotal === 2900) {
        // 29€ = Plus
        plan = 'plus';
      } else if (amountTotal === 7900) {
        // 79€ = Pro
        plan = 'pro';
      }

      // Si on n'a pas pu déterminer le plan par le montant, essayer de récupérer les détails de la session
      if (!plan && stripe) {
        try {
          const sessionDetails = await stripe.checkout.sessions.retrieve(session.id, {
            expand: ['line_items'],
          });
          
          const lineItems = sessionDetails.line_items?.data;
          if (lineItems && lineItems.length > 0) {
            const priceId = lineItems[0].price?.id;
            const amount = lineItems[0].amount_total;
            
            // Déterminer le plan par le montant
            if (amount === 2900) {
              plan = 'plus';
            } else if (amount === 7900) {
              plan = 'pro';
            }
            
            // Alternative : utiliser l'URL de la session pour déterminer le plan
            if (!plan && sessionDetails.url) {
              if (sessionDetails.url.includes('6oUfZh8dFeSC3UbcG32VG00')) {
                plan = 'plus';
              } else if (sessionDetails.url.includes('9B6dR951t6m6aizfSf2VG01')) {
                plan = 'pro';
              }
            }
          }
        } catch (err) {
          console.error('Erreur récupération détails session:', err);
        }
      }

      if (plan) {
        // Mettre à jour le plan de l'utilisateur
        const { error } = await admin.auth.admin.updateUserById(user.id, {
          user_metadata: {
            ...user.user_metadata,
            plan,
            plan_updated_at: new Date().toISOString(),
            stripe_customer_id: session.customer,
            stripe_subscription_id: session.subscription,
            stripe_checkout_session_id: session.id,
          },
        });

        if (error) {
          console.error('Erreur mise à jour plan:', error);
        } else {
          console.log(`Plan ${plan} activé pour ${customerEmail}`);
        }
      } else {
        console.warn(`Impossible de déterminer le plan pour la session ${session.id}`);
      }
      break;
    }

    case 'customer.subscription.updated':
    case 'customer.subscription.deleted': {
      const subscription = event.data.object as any;
      const customerId = subscription.customer;

      // Récupérer les détails du customer pour obtenir l'email
      if (stripe && customerId) {
        try {
          const customer = await stripe.customers.retrieve(customerId);
          if (typeof customer !== 'string' && 'email' in customer && customer.email) {
            // Chercher l'utilisateur par email
            const { data: users } = await admin.auth.admin.listUsers();
            const user = users.users.find((u) => u.email === customer.email);

            if (user) {
              if (event.type === 'customer.subscription.deleted') {
                // Rétrograder vers Basic si l'abonnement est annulé
                await admin.auth.admin.updateUserById(user.id, {
                  user_metadata: {
                    ...user.user_metadata,
                    plan: 'basic',
                    plan_updated_at: new Date().toISOString(),
                  },
                });
                console.log(`Abonnement annulé, plan Basic activé pour ${customer.email}`);
              }
            }
          }
        } catch (err) {
          console.error('Erreur gestion subscription:', err);
        }
      }
      break;
    }

    default:
      console.log(`Événement non géré: ${event.type}`);
  }

  return NextResponse.json({ received: true });
}

