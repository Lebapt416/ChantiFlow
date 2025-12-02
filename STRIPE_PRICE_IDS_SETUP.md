# Configuration des Price IDs Stripe pour ChantiFlow

## Migration vers le SDK Stripe

Le système de paiement a été migré pour utiliser le SDK Stripe officiel (`stripe.checkout.sessions.create`) au lieu des liens de paiement directs. Cela offre une meilleure sécurité, traçabilité et flexibilité.

## Variables d'environnement requises

Ajoutez ces variables dans votre fichier `.env.local` (ou dans les variables d'environnement de votre plateforme de déploiement) :

```env
# Clé secrète Stripe (obligatoire)
STRIPE_SECRET_KEY=sk_test_...

# Price IDs Stripe (obligatoires pour utiliser le SDK)
STRIPE_PRICE_ID_PLUS_MONTHLY=price_...
STRIPE_PRICE_ID_PLUS_ANNUAL=price_...
STRIPE_PRICE_ID_PRO_MONTHLY=price_...
STRIPE_PRICE_ID_PRO_ANNUAL=price_...

# URL de base de l'application (pour les redirections)
NEXT_PUBLIC_APP_BASE_URL=https://votre-domaine.com
```

## Comment créer les Price IDs dans Stripe

### 1. Créer les produits dans Stripe Dashboard

1. Allez sur [dashboard.stripe.com](https://dashboard.stripe.com)
2. Naviguez vers **Products** > **Add product**
3. Créez 4 produits :
   - **ChantiFlow Plus (Mensuel)** : 29€/mois, récurrent
   - **ChantiFlow Plus (Annuel)** : 29€ × 12 = 348€/an, récurrent
   - **ChantiFlow Pro (Mensuel)** : 79€/mois, récurrent
   - **ChantiFlow Pro (Annuel)** : 79€ × 12 = 948€/an, récurrent

### 2. Récupérer les Price IDs

Pour chaque produit créé :
1. Cliquez sur le produit
2. Dans la section **Pricing**, vous verrez le **Price ID** (commence par `price_...`)
3. Copiez ce Price ID et ajoutez-le dans vos variables d'environnement

**Exemple de configuration :**
```env
STRIPE_PRICE_ID_PLUS_MONTHLY=price_1ABC123def456GHI
STRIPE_PRICE_ID_PLUS_ANNUAL=price_1XYZ789abc012JKL
STRIPE_PRICE_ID_PRO_MONTHLY=price_1MNO345pqr678STU
STRIPE_PRICE_ID_PRO_ANNUAL=price_1VWX901yza234BCD
```

## Fonctionnement

### Avec Price IDs configurés (recommandé)

Si tous les Price IDs sont configurés, le système utilise `stripe.checkout.sessions.create` :
- ✅ Session de checkout créée dynamiquement
- ✅ Metadata automatique (userId, plan, billing_period)
- ✅ Customer email pré-rempli
- ✅ Meilleure traçabilité pour le webhook
- ✅ Support des codes promotionnels

### Sans Price IDs (fallback)

Si les Price IDs ne sont pas configurés, le système utilise les anciens liens de paiement directs :
- ⚠️ Moins sécurisé (pas de metadata automatique)
- ⚠️ Nécessite de parser l'email dans le webhook
- ⚠️ Pas de support des codes promotionnels

## Avantages de la migration

1. **Sécurité** : Les sessions sont créées côté serveur avec validation
2. **Traçabilité** : Metadata automatique (userId, plan) pour le webhook
3. **Flexibilité** : Support des codes promotionnels, modes de paiement multiples
4. **Maintenance** : Plus besoin de gérer des liens statiques

## Test

1. Configurez les Price IDs dans vos variables d'environnement
2. Redémarrez votre serveur de développement
3. Testez un checkout :
   - Le système devrait créer une session Stripe
   - Vérifiez dans le dashboard Stripe que la session contient les metadata

## Migration depuis les anciens liens

Si vous utilisez déjà les anciens liens de paiement :
1. Créez les produits et Price IDs dans Stripe
2. Ajoutez les variables d'environnement
3. Le système basculera automatiquement vers le SDK
4. Les anciens liens continueront de fonctionner (fallback)

## Support

En cas de problème :
1. Vérifiez que tous les Price IDs sont correctement configurés
2. Vérifiez que `STRIPE_SECRET_KEY` est valide
3. Consultez les logs du serveur pour voir les erreurs détaillées

