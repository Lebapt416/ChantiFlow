# Configuration Stripe sur Vercel - Guide Rapide

## ✅ Clés Stripe configurées localement

Les clés Stripe ont été ajoutées dans votre fichier `.env.local` (qui est ignoré par Git).

## 🚀 Configuration sur Vercel (Production)

### Étape 1 : Ajouter les variables d'environnement

1. Allez sur [vercel.com](https://vercel.com)
2. Sélectionnez votre projet **ChantiFlow**
3. Allez dans **Settings** > **Environment Variables**
4. Ajoutez les variables suivantes :

#### Variable 1 : Clé secrète Stripe
- **Name** : `STRIPE_SECRET_KEY`
- **Value** : `sk_live_...` (votre clé secrète Stripe - commence par `sk_live_` ou `sk_test_`)
- **Environments** : ✅ Production, ✅ Preview, ✅ Development

#### Variable 2 : Clé publique Stripe (optionnel, pour usage côté client)
- **Name** : `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY`
- **Value** : `pk_live_...` (votre clé publique Stripe - commence par `pk_live_` ou `pk_test_`)
- **Environments** : ✅ Production, ✅ Preview, ✅ Development

### Étape 2 : Configurer les Price IDs (Recommandé)

Pour utiliser le nouveau système de paiement avec le SDK Stripe, vous devez créer les Price IDs dans Stripe Dashboard :

1. Allez sur [dashboard.stripe.com](https://dashboard.stripe.com)
2. Créez 4 produits avec leurs prix :
   - **ChantiFlow Plus (Mensuel)** : 29€/mois
   - **ChantiFlow Plus (Annuel)** : 348€/an
   - **ChantiFlow Pro (Mensuel)** : 79€/mois
   - **ChantiFlow Pro (Annuel)** : 948€/an
3. Récupérez les Price IDs (commencent par `price_...`)
4. Ajoutez-les dans Vercel :
   - `STRIPE_PRICE_ID_PLUS_MONTHLY`
   - `STRIPE_PRICE_ID_PLUS_ANNUAL`
   - `STRIPE_PRICE_ID_PRO_MONTHLY`
   - `STRIPE_PRICE_ID_PRO_ANNUAL`

**Note** : Si vous ne configurez pas les Price IDs, le système utilisera automatiquement les anciens liens de paiement (fallback).

### Étape 3 : Configurer le Webhook Stripe

1. Allez sur [dashboard.stripe.com](https://dashboard.stripe.com) > **Developers** > **Webhooks**
2. Cliquez sur **Add endpoint**
3. **URL** : `https://votre-domaine.vercel.app/api/stripe/webhook`
4. **Événements à écouter** :
   - `checkout.session.completed`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
5. Copiez le **Signing secret** (commence par `whsec_...`)
6. Ajoutez-le dans Vercel :
   - **Name** : `STRIPE_WEBHOOK_SECRET`
   - **Value** : `whsec_...`
   - **Environments** : ✅ Production

### Étape 4 : Redéployer

Après avoir ajouté toutes les variables :
1. Allez dans **Deployments**
2. Cliquez sur les **3 points** du dernier déploiement
3. Sélectionnez **Redeploy**
4. Ou faites un nouveau commit pour déclencher un redéploiement automatique

## ✅ Vérification

Pour vérifier que tout fonctionne :

1. Testez un paiement sur votre site
2. Vérifiez dans Stripe Dashboard > **Payments** que les paiements arrivent
3. Vérifiez dans Stripe Dashboard > **Webhooks** que les événements sont bien reçus

## 🔒 Sécurité

- ✅ Les clés sont stockées de manière sécurisée dans Vercel
- ✅ Les clés ne sont jamais exposées dans le code source
- ✅ Les clés sont chiffrées dans Vercel
- ⚠️ Ne partagez jamais ces clés publiquement

## 📚 Documentation

- Voir `STRIPE_PRICE_IDS_SETUP.md` pour configurer les Price IDs
- Voir `ENV_SETUP.md` pour la liste complète des variables d'environnement

