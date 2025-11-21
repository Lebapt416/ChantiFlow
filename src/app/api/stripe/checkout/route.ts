import { NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { stripe, STRIPE_PRICE_IDS, isAdminUser } from '@/lib/stripe';

export async function POST(request: Request) {
  try {
    const { plan } = await request.json();

    if (!plan || !['plus', 'pro'].includes(plan)) {
      return NextResponse.json({ error: 'Plan invalide' }, { status: 400 });
    }

    // Vérifier l'authentification
    const supabase = await createSupabaseServerClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user || !user.email) {
      return NextResponse.json({ error: 'Non authentifié' }, { status: 401 });
    }

    // Si c'est l'admin, pas besoin de Stripe
    if (isAdminUser(user.email)) {
      return NextResponse.json({ 
        error: 'Les paiements sont désactivés pour ce compte',
        skipPayment: true 
      }, { status: 200 });
    }

    // Vérifier que Stripe est configuré
    if (!stripe) {
      return NextResponse.json(
        { error: 'Stripe non configuré. Vérifiez STRIPE_SECRET_KEY.' },
        { status: 500 }
      );
    }

    const priceId = STRIPE_PRICE_IDS[plan as 'plus' | 'pro'];

    // Créer la session de checkout Stripe
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
        plan,
      },
      success_url: `${process.env.NEXT_PUBLIC_APP_BASE_URL || 'http://localhost:3000'}/account/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${process.env.NEXT_PUBLIC_APP_BASE_URL || 'http://localhost:3000'}/account/cancel`,
    });

    return NextResponse.json({ url: session.url });
  } catch (error) {
    console.error('Erreur création checkout Stripe:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Erreur inconnue' },
      { status: 500 }
    );
  }
}

