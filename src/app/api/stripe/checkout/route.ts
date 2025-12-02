import { NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { stripe, isAdminUser, STRIPE_PRICE_IDS, STRIPE_CHECKOUT_LINKS, hasStripePriceIds } from '@/lib/stripe';
import { logger } from '@/lib/logger';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { plan, isAnnual = false } = body;

    logger.info('Requête checkout Stripe reçue', {
      plan,
      isAnnual,
    });

    if (!plan || !['plus', 'pro'].includes(plan)) {
      logger.error('Plan invalide reçu', { plan });
      return NextResponse.json({ error: 'Plan invalide' }, { status: 400 });
    }

    // Vérifier l'authentification
    const supabase = await createSupabaseServerClient();
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser();

    if (authError) {
      logger.error('Erreur authentification lors du checkout', { error: authError.message });
      return NextResponse.json({ error: 'Erreur d\'authentification' }, { status: 401 });
    }

    if (!user || !user.email) {
      logger.error('Utilisateur non authentifié lors du checkout');
      return NextResponse.json({ error: 'Non authentifié' }, { status: 401 });
    }

    logger.info('Checkout initié pour utilisateur', {
      userId: user.id,
      email: user.email,
      plan,
      isAnnual,
    });

    // Si c'est l'admin, pas besoin de Stripe
    if (isAdminUser(user.email)) {
      logger.info('Compte admin détecté, paiement désactivé', { email: user.email });
      return NextResponse.json({ 
        error: 'Les paiements sont désactivés pour ce compte',
        skipPayment: true 
      }, { status: 200 });
    }

    // Vérifier si Stripe est configuré
    if (!stripe) {
      logger.error('Stripe non configuré - STRIPE_SECRET_KEY manquante');
      return NextResponse.json({ error: 'Service de paiement non configuré' }, { status: 500 });
    }

    // Utiliser le SDK Stripe si les Price IDs sont configurés, sinon fallback sur les liens
    if (hasStripePriceIds()) {
      try {
        const priceId = isAnnual 
          ? STRIPE_PRICE_IDS[plan as 'plus' | 'pro'].annual
          : STRIPE_PRICE_IDS[plan as 'plus' | 'pro'].monthly;

        if (!priceId) {
          logger.error('Price ID manquant pour le plan', { plan, isAnnual });
          return NextResponse.json({ error: 'Configuration de paiement invalide' }, { status: 500 });
        }

        // Récupérer l'URL de base de l'application
        const baseUrl = process.env.NEXT_PUBLIC_APP_BASE_URL || 
                       process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : 
                       'http://localhost:3000';

        // Créer une session de checkout avec le SDK Stripe
        const session = await stripe.checkout.sessions.create({
          mode: 'subscription',
          payment_method_types: ['card'],
          line_items: [
            {
              price: priceId,
              quantity: 1,
            },
          ],
          customer_email: user.email,
          metadata: {
            userId: user.id,
            plan: plan,
            billing_period: isAnnual ? 'annual' : 'monthly',
          },
          success_url: `${baseUrl}/account/success?session_id={CHECKOUT_SESSION_ID}`,
          cancel_url: `${baseUrl}/account/cancel`,
          allow_promotion_codes: true,
        });

        logger.info('Session Stripe checkout créée avec succès', {
          sessionId: session.id,
          userId: user.id,
          plan,
          isAnnual,
        });

        return NextResponse.json({ url: session.url });
      } catch (stripeError) {
        logger.error('Erreur lors de la création de la session Stripe', {
          error: stripeError instanceof Error ? stripeError.message : 'Erreur inconnue',
          userId: user.id,
        });
        return NextResponse.json(
          { error: 'Erreur lors de la création de la session de paiement' },
          { status: 500 }
        );
      }
    } else {
      // Fallback : utiliser les liens de paiement directs (ancien système)
      logger.warn('Price IDs non configurés, utilisation des liens de paiement directs', {
        userId: user.id,
      });

      const checkoutLink = isAnnual 
        ? STRIPE_CHECKOUT_LINKS[plan as 'plus' | 'pro'].annual
        : STRIPE_CHECKOUT_LINKS[plan as 'plus' | 'pro'].monthly;

      if (!checkoutLink) {
        logger.error('Lien checkout non trouvé', { plan, isAnnual });
        return NextResponse.json({ error: 'Lien de paiement non configuré' }, { status: 500 });
      }

      // Ajouter l'email comme paramètre pour faciliter l'identification
      const urlWithEmail = `${checkoutLink}?prefilled_email=${encodeURIComponent(user.email)}`;

      logger.info('Redirection vers lien Stripe direct', {
        userId: user.id,
        isAnnual,
      });

      return NextResponse.json({ url: urlWithEmail });
    }
  } catch (error) {
    logger.error('Erreur lors du traitement du checkout Stripe', {
      error: error instanceof Error ? error.message : 'Erreur inconnue',
      stack: error instanceof Error ? error.stack : undefined,
    });
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Erreur inconnue' },
      { status: 500 }
    );
  }
}

