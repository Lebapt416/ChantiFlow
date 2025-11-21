# Configuration Stripe pour ChantiFlow

## Variables d'environnement requises

Ajoutez ces variables dans votre fichier `.env.local` :

```env
# Clé secrète Stripe (trouvable dans le dashboard Stripe > Developers > API keys)
STRIPE_SECRET_KEY=sk_test_...

# Secret du webhook Stripe (trouvable après création du webhook)
STRIPE_WEBHOOK_SECRET=whsec_...

# URL de base de l'application (pour les redirections)
NEXT_PUBLIC_APP_BASE_URL=https://votre-domaine.com
```

## Liens de paiement Stripe Checkout

Les liens de paiement sont déjà configurés dans le code :
- **ChantiFlow Plus** : `https://buy.stripe.com/6oUfZh8dFeSC3UbcG32VG00` (29€/mois)
- **ChantiFlow Pro** : `https://buy.stripe.com/9B6dR951t6m6aizfSf2VG01` (79€/mois)

## Étapes de configuration

### 1. Créer un compte Stripe

1. Allez sur [stripe.com](https://stripe.com)
2. Créez un compte (mode test pour commencer)
3. Récupérez votre clé secrète dans **Developers > API keys**

### 2. Configurer les liens de paiement

Les liens Stripe Checkout sont déjà créés. Vous pouvez les personnaliser dans le dashboard Stripe :
1. Allez dans **Products** > **Payment Links**
2. Configurez les redirections après paiement :
   - **Success URL** : `https://votre-domaine.com/account/success`
   - **Cancel URL** : `https://votre-domaine.com/account/cancel`

### 3. Configurer le webhook

**IMPORTANT** : Le webhook est essentiel pour activer automatiquement le plan après paiement.

1. Allez dans **Developers > Webhooks**
2. Cliquez sur **Add endpoint**
3. URL : `https://votre-domaine.com/api/stripe/webhook`
4. Événements à écouter :
   - `checkout.session.completed` (essentiel pour activer le plan)
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
5. Copiez le **Signing secret** (commence par `whsec_...`)
6. Ajoutez-le dans vos variables d'environnement

### 4. Compte admin

Le compte `admin@gmail.com` peut changer de plan **sans payer**. Les paiements sont automatiquement désactivés pour ce compte.

## Fonctionnement du système

### Flux de paiement

1. **Utilisateur clique sur "Payer et activer"** → Redirection vers le lien Stripe Checkout
2. **Utilisateur paie** → Stripe traite le paiement
3. **Stripe envoie un webhook** → `checkout.session.completed`
4. **Webhook détecte l'email** → Trouve l'utilisateur dans Supabase
5. **Webhook détermine le plan** → Selon le montant payé (29€ = Plus, 79€ = Pro)
6. **Plan mis à jour automatiquement** → L'utilisateur a accès aux fonctionnalités

### Détection du plan

Le webhook détermine automatiquement le plan en fonction du montant payé :
- **29€** = Plan Plus
- **79€** = Plan Pro

Si vous modifiez les prix dans Stripe, mettez à jour les montants dans `/src/app/api/stripe/webhook/route.ts`.

## Test en mode développement

1. Utilisez les clés de test Stripe (commencent par `sk_test_`)
2. Utilisez les cartes de test Stripe :
   - Succès : `4242 4242 4242 4242`
   - Échec : `4000 0000 0000 0002`
   - Date : n'importe quelle date future
   - CVC : n'importe quel 3 chiffres

## Déploiement

1. Passez en mode production dans Stripe
2. Récupérez les clés de production (commencent par `sk_live_`)
3. Mettez à jour les variables d'environnement sur Vercel
4. Configurez le webhook avec l'URL de production

