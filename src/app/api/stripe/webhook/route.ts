import { NextResponse } from 'next/server';
import { headers } from 'next/headers';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';
import { stripe } from '@/lib/stripe';
import { sendAccountCreatedEmail } from '@/lib/email';

type PlanKey = 'plus' | 'pro';
type AddOnKey = 'extra_workers' | 'extra_sites';

function normalizePlanKey(value?: string | null): PlanKey | null {
  if (!value) return null;
  const normalized = value.toLowerCase();
  return normalized === 'plus' || normalized === 'pro' ? normalized : null;
}

function normalizeAddOnKey(value?: string | null): AddOnKey | null {
  if (!value) return null;
  const normalized = value.toLowerCase();
  if (normalized === 'extra_workers' || normalized === 'extra_sites') {
    return normalized;
  }
  return null;
}

// Fonction pour générer un mot de passe sécurisé
function generateSecurePassword(): string {
  const length = 16;
  const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*';
  let password = '';
  
  // S'assurer d'avoir au moins un caractère de chaque type
  password += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[Math.floor(Math.random() * 26)]; // Majuscule
  password += 'abcdefghijklmnopqrstuvwxyz'[Math.floor(Math.random() * 26)]; // Minuscule
  password += '0123456789'[Math.floor(Math.random() * 10)]; // Chiffre
  password += '!@#$%^&*'[Math.floor(Math.random() * 8)]; // Spécial
  
  // Remplir le reste
  for (let i = password.length; i < length; i++) {
    password += charset[Math.floor(Math.random() * charset.length)];
  }
  
  // Mélanger les caractères
  return password.split('').sort(() => Math.random() - 0.5).join('');
}

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
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
        // Vous devrez mapper vos price IDs ici si nécessaire
        // Pour l'instant, on va utiliser une autre méthode
      }

      // Alternative : utiliser l'URL de la session ou les métadonnées du produit
      // Pour les liens buy.stripe.com, on peut vérifier le customer_email
      // et chercher l'utilisateur correspondant dans Supabase
      
      // Chercher l'utilisateur par email
      const { data: users } = await admin.auth.admin.listUsers();
      let user = users.users.find((u) => u.email === customerEmail);

      // Si l'utilisateur n'existe pas, créer un compte automatiquement
      if (!user) {
        console.log(`📝 Création automatique du compte pour ${customerEmail}`);
        
        // Générer un mot de passe aléatoire sécurisé
        const randomPassword = generateSecurePassword();
        
        try {
          // Créer l'utilisateur avec l'admin client
          const { data: newUser, error: createError } = await admin.auth.admin.createUser({
            email: customerEmail,
            password: randomPassword,
            email_confirm: true, // Confirmer l'email automatiquement
            user_metadata: {
              created_via: 'stripe_payment',
              created_at: new Date().toISOString(),
            },
          });

          if (createError || !newUser) {
            console.error('❌ Erreur création compte:', createError);
            // Continuer quand même pour essayer de déterminer le plan
          } else {
            console.log(`✅ Compte créé avec succès pour ${customerEmail}`);
            user = newUser.user;
            
            // Envoyer un email avec les identifiants
            try {
              await sendAccountCreatedEmail({
                userEmail: customerEmail,
                temporaryPassword: randomPassword,
              });
            } catch (emailError) {
              console.error('❌ Erreur envoi email identifiants:', emailError);
              // Ne pas bloquer le processus si l'email échoue
            }
          }
        } catch (error) {
          console.error('❌ Exception lors de la création du compte:', error);
          // Continuer pour essayer de déterminer le plan quand même
        }
      }

      if (!user) {
        console.error(`❌ Impossible de créer ou trouver l'utilisateur pour ${customerEmail}`);
        break;
      }

      // Priorité : utiliser les métadonnées Stripe si disponibles
      // ⚠️ Ajoutez la métadonnée plan_key (= plus | pro) sur vos Liens de Paiement Stripe
      let planMetadata = normalizePlanKey(session.metadata?.plan_key as string | undefined);
      if (planMetadata) {
        plan = planMetadata;
      }
      // ⚠️ Ajoutez aussi la métadonnée addon_key (= extra_workers | extra_sites) sur vos Liens Stripe
      let addOnType = normalizeAddOnKey(session.metadata?.addon_key as string | undefined);

      // Fallback sécurisé : déterminer plan/add-on par le montant
      const amountTotal = session.amount_total; // en centimes
      if (!plan) {
        if (amountTotal === 2900) {
          plan = 'plus';
        } else if (amountTotal === 7900) {
          plan = 'pro';
        }
      }
      if (!addOnType) {
        if (amountTotal === 1000) {
          addOnType = 'extra_workers';
        } else if (amountTotal === 500) {
          addOnType = 'extra_sites';
        }
      }

      // Si on n'a pas pu déterminer le plan/add-on, récupérer les détails de la session
      if ((!plan || !addOnType) && stripe) {
        try {
          const sessionDetails = await stripe.checkout.sessions.retrieve(session.id, {
            expand: ['line_items'],
          });

          if (!plan) {
            planMetadata = normalizePlanKey(sessionDetails.metadata?.plan_key as string | undefined);
            if (planMetadata) {
              plan = planMetadata;
            }
          }
          if (!addOnType) {
            addOnType = normalizeAddOnKey(sessionDetails.metadata?.addon_key as string | undefined);
          }
          
          const lineItems = sessionDetails.line_items?.data;
          if (lineItems && lineItems.length > 0) {
            const amount = lineItems[0].amount_total;
            
            if (!plan) {
              if (amount === 2900) {
                plan = 'plus';
              } else if (amount === 7900) {
                plan = 'pro';
              }
            }

            if (!addOnType) {
              if (amount === 1000) {
                addOnType = 'extra_workers';
              } else if (amount === 500) {
                addOnType = 'extra_sites';
              }
            }
            
            // Alternative : utiliser l'URL de la session pour déterminer le plan
            if (!plan && sessionDetails.url) {
              // Liens mensuels
              if (sessionDetails.url.includes('6oUfZh8dFeSC3UbcG32VG00')) {
                plan = 'plus';
              } else if (sessionDetails.url.includes('9B6dR951t6m6aizfSf2VG01')) {
                plan = 'pro';
              }
              // Liens annuels
              else if (sessionDetails.url.includes('aFa3cv79BaCmbmD49x2VG04')) {
                plan = 'plus';
              } else if (sessionDetails.url.includes('cNibJ1alN9yi62j6hF2VG05')) {
                plan = 'pro';
              }
            }
          }
        } catch (err) {
          console.error('Erreur récupération détails session:', err);
        }
      }

      const isAddOn = !!addOnType;
      
      if (isAddOn && addOnType) {
        // C'est un add-on
        const currentMetadata = user.user_metadata || {};
        const currentAddOns = currentMetadata.addOns || { extra_workers: 0, extra_sites: 0 };
        
        const updatedAddOns = { ...currentAddOns };
        
        if (addOnType === 'extra_workers') {
          // Add-on +5 employés (10€)
          updatedAddOns.extra_workers = (updatedAddOns.extra_workers || 0) + 1;
          console.log(`Add-on +5 employés acheté pour ${customerEmail}`);
        } else if (addOnType === 'extra_sites') {
          // Add-on +2 chantiers (5€)
          updatedAddOns.extra_sites = (updatedAddOns.extra_sites || 0) + 1;
          console.log(`Add-on +2 chantiers acheté pour ${customerEmail}`);
        }
        
        // Mettre à jour les add-ons de l'utilisateur
        const { error } = await admin.auth.admin.updateUserById(user.id, {
          user_metadata: {
            ...currentMetadata,
            addOns: updatedAddOns,
            addOns_updated_at: new Date().toISOString(),
            stripe_customer_id: session.customer,
            stripe_subscription_id: session.subscription,
            stripe_checkout_session_id: session.id,
          },
        });

        if (error) {
          console.error('Erreur mise à jour add-ons:', error);
        } else {
          console.log(`Add-ons mis à jour pour ${customerEmail}:`, updatedAddOns);
        }
      } else if (plan) {
        // C'est un plan
        // Déterminer si c'est annuel (vérifier l'URL de la session)
        let isAnnualPayment = false;
        if (session.url) {
          // Liens annuels
          if (session.url.includes('aFa3cv79BaCmbmD49x2VG04') || session.url.includes('cNibJ1alN9yi62j6hF2VG05')) {
            isAnnualPayment = true;
          }
        }
        // Vérifier aussi dans les métadonnées
        if (session.metadata?.is_annual === 'true' || session.metadata?.is_annual === '1') {
          isAnnualPayment = true;
        }
        
        // Calculer la date d'expiration
        const now = new Date();
        const expirationDate = isAnnualPayment 
          ? new Date(now.getFullYear() + 1, now.getMonth(), now.getDate())
          : null; // Les paiements mensuels n'ont pas d'expiration (abonnement récurrent)
        
        // Mettre à jour le plan de l'utilisateur
        const metadataUpdate: Record<string, unknown> = {
          ...user.user_metadata,
          plan,
          plan_updated_at: new Date().toISOString(),
          stripe_customer_id: session.customer,
          stripe_subscription_id: session.subscription,
          stripe_checkout_session_id: session.id,
        };
        
        if (isAnnualPayment && expirationDate) {
          metadataUpdate.plan_expires_at = expirationDate.toISOString();
          metadataUpdate.plan_type = 'annual';
          console.log(`Plan ${plan} annuel activé pour ${customerEmail}, expire le ${expirationDate.toISOString()}`);
        } else {
          metadataUpdate.plan_type = 'monthly';
          console.log(`Plan ${plan} mensuel activé pour ${customerEmail}`);
        }
        
        const { error } = await admin.auth.admin.updateUserById(user.id, {
          user_metadata: metadataUpdate,
        });

        if (error) {
          console.error('Erreur mise à jour plan:', error);
        } else {
          console.log(`Plan ${plan} ${isAnnualPayment ? 'annuel' : 'mensuel'} activé pour ${customerEmail}`);
        }
      } else {
        console.warn(`Impossible de déterminer le plan ou add-on pour la session ${session.id}`);
      }
      break;
    }

    case 'customer.subscription.updated':
    case 'customer.subscription.deleted': {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
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

