# Configuration Stripe pour ChantiFlow

## Variables d'environnement requises

Ajoutez ces variables dans votre fichier `.env.local` :

```env
# Clé secrète Stripe (trouvable dans le dashboard Stripe > Developers > API keys)
STRIPE_SECRET_KEY=sk_test_...

# ID des prix Stripe (créer dans Stripe > Products > Pricing)
STRIPE_PRICE_ID_PLUS=price_...
STRIPE_PRICE_ID_PRO=price_...

# Secret du webhook Stripe (trouvable après création du webhook)
STRIPE_WEBHOOK_SECRET=whsec_...

# URL de base de l'application (pour les redirections)
NEXT_PUBLIC_APP_BASE_URL=https://votre-domaine.com
```

## Étapes de configuration

### 1. Créer un compte Stripe

1. Allez sur [stripe.com](https://stripe.com)
2. Créez un compte (mode test pour commencer)
3. Récupérez votre clé secrète dans **Developers > API keys**

### 2. Créer les produits et prix

1. Allez dans **Products** dans le dashboard Stripe
2. Créez deux produits :
   - **ChantiFlow Plus** (29€/mois)
   - **ChantiFlow Pro** (79€/mois)
3. Pour chaque produit, créez un prix récurrent mensuel
4. Copiez les **Price ID** (commencent par `price_...`)

### 3. Configurer le webhook

1. Allez dans **Developers > Webhooks**
2. Cliquez sur **Add endpoint**
3. URL : `https://votre-domaine.com/api/stripe/webhook`
4. Événements à écouter :
   - `checkout.session.completed`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
5. Copiez le **Signing secret** (commence par `whsec_...`)

### 4. Compte admin

Le compte `admin@gmail.com` peut changer de plan **sans payer**. Les paiements sont automatiquement désactivés pour ce compte.

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

