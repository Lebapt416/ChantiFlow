import { NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { STRIPE_CHECKOUT_LINKS, isAdminUser } from '@/lib/stripe';

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

    // Rediriger vers le lien Stripe Checkout direct
    const checkoutLink = STRIPE_CHECKOUT_LINKS[plan as 'plus' | 'pro'];
    
    // Ajouter l'email comme paramètre pour faciliter l'identification
    const urlWithEmail = `${checkoutLink}?prefilled_email=${encodeURIComponent(user.email)}`;

    return NextResponse.json({ url: urlWithEmail });
  } catch (error) {
    console.error('Erreur redirection checkout Stripe:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Erreur inconnue' },
      { status: 500 }
    );
  }
}

