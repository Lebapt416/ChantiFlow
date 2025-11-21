# Configuration des liens Stripe Checkout

## Liens de paiement configurés

- **ChantiFlow Plus** : https://buy.stripe.com/6oUfZh8dFeSC3UbcG32VG00 (29€/mois)
- **ChantiFlow Pro** : https://buy.stripe.com/9B6dR951t6m6aizfSf2VG01 (79€/mois)

## Configuration dans Stripe Dashboard

### 1. Configurer les redirections après paiement

Pour chaque lien de paiement dans Stripe :

1. Allez dans **Products** > **Payment Links**
2. Cliquez sur le lien correspondant
3. Dans **After payment**, configurez :
   - **Success page URL** : `https://votre-domaine.com/account/success`
   - **Cancel page URL** : `https://votre-domaine.com/account/cancel`

### 2. Activer la collecte d'email

Assurez-vous que la collecte d'email est activée dans les paramètres du Payment Link :
- **Customer information** > **Email** : Activé (requis)

Cela permet au webhook d'identifier l'utilisateur après le paiement.

### 3. Vérifier le webhook

Le webhook doit être configuré pour écouter l'événement `checkout.session.completed`.

URL du webhook : `https://votre-domaine.com/api/stripe/webhook`

## Fonctionnement

1. **Utilisateur clique sur "Payer et activer"** → Redirection vers le lien Stripe Checkout
2. **Utilisateur paie** → Stripe traite le paiement
3. **Stripe envoie un webhook** → `checkout.session.completed` avec l'email du customer
4. **Webhook trouve l'utilisateur** → Par email dans Supabase
5. **Webhook détermine le plan** → Selon le montant (29€ = Plus, 79€ = Pro)
6. **Plan mis à jour** → L'utilisateur a accès aux fonctionnalités

## Test

1. Utilisez une carte de test : `4242 4242 4242 4242`
2. Utilisez l'email d'un compte existant dans votre application
3. Après le paiement, vérifiez que le plan est mis à jour dans `/account`

