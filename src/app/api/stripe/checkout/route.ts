import { NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { STRIPE_CHECKOUT_LINKS, isAdminUser } from '@/lib/stripe';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { plan } = body;

    console.log('📥 Requête checkout reçue pour plan:', plan);

    if (!plan || !['plus', 'pro'].includes(plan)) {
      console.error('❌ Plan invalide:', plan);
      return NextResponse.json({ error: 'Plan invalide' }, { status: 400 });
    }

    // Vérifier l'authentification
    const supabase = await createSupabaseServerClient();
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser();

    if (authError) {
      console.error('❌ Erreur auth:', authError);
      return NextResponse.json({ error: 'Erreur d\'authentification' }, { status: 401 });
    }

    if (!user || !user.email) {
      console.error('❌ Utilisateur non trouvé');
      return NextResponse.json({ error: 'Non authentifié' }, { status: 401 });
    }

    console.log('👤 Utilisateur:', user.email);

    // Si c'est l'admin, pas besoin de Stripe
    if (isAdminUser(user.email)) {
      console.log('🔑 Compte admin détecté, paiement désactivé');
      return NextResponse.json({ 
        error: 'Les paiements sont désactivés pour ce compte',
        skipPayment: true 
      }, { status: 200 });
    }

    // Rediriger vers le lien Stripe Checkout direct
    const checkoutLink = STRIPE_CHECKOUT_LINKS[plan as 'plus' | 'pro'];
    
    if (!checkoutLink) {
      console.error('❌ Lien checkout non trouvé pour plan:', plan);
      return NextResponse.json({ error: 'Lien de paiement non configuré' }, { status: 500 });
    }

    // Ajouter l'email comme paramètre pour faciliter l'identification
    const urlWithEmail = `${checkoutLink}?prefilled_email=${encodeURIComponent(user.email)}`;

    console.log('✅ Redirection vers:', urlWithEmail);

    return NextResponse.json({ url: urlWithEmail });
  } catch (error) {
    console.error('❌ Erreur redirection checkout Stripe:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Erreur inconnue' },
      { status: 500 }
    );
  }
}

