# Configuration des Variables d'Environnement - ChantiFlow

## ⚠️ SÉCURITÉ IMPORTANTE

**NE JAMAIS COMMITER** les fichiers `.env.local` ou `.env` contenant des clés secrètes dans Git.

## Variables d'environnement requises

### Stripe (Production)

```env
# Clé secrète Stripe (Production)
STRIPE_SECRET_KEY=sk_live_...

# Clé publique Stripe (Production) - optionnelle, utilisée côté client si nécessaire
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_live_...

# Price IDs Stripe (à créer dans le dashboard Stripe)
STRIPE_PRICE_ID_PLUS_MONTHLY=price_...
STRIPE_PRICE_ID_PLUS_ANNUAL=price_...
STRIPE_PRICE_ID_PRO_MONTHLY=price_...
STRIPE_PRICE_ID_PRO_ANNUAL=price_...

# Secret du webhook Stripe (trouvable dans Stripe Dashboard > Webhooks)
STRIPE_WEBHOOK_SECRET=whsec_...
```

### Supabase

```env
# URL Supabase
NEXT_PUBLIC_SUPABASE_URL=https://votre-projet.supabase.co

# Clé anonyme Supabase
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Clé de service Supabase (pour les opérations admin)
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Autres services

```env
# URL de base de l'application
NEXT_PUBLIC_APP_BASE_URL=https://votre-domaine.com

# API de prédiction IA (optionnel)
NEXT_PUBLIC_PREDICTION_API_URL=https://votre-api.com

# Resend (pour les emails)
RESEND_API_KEY=re_...

# Sentry (optionnel, pour le monitoring)
NEXT_PUBLIC_SENTRY_DSN=https://...@sentry.io/...
```

## Configuration locale

1. Créez un fichier `.env.local` à la racine du projet
2. Ajoutez toutes les variables ci-dessus avec vos vraies valeurs
3. Le fichier `.env.local` est automatiquement ignoré par Git

## Configuration sur Vercel (Production)

1. Allez sur [vercel.com](https://vercel.com) > Votre projet > Settings > Environment Variables
2. Ajoutez toutes les variables d'environnement une par une
3. Pour chaque variable, sélectionnez les environnements (Production, Preview, Development)
4. Redéployez l'application après avoir ajouté les variables

## Vérification

Pour vérifier que les variables sont bien configurées :

```bash
# En local
cat .env.local

# Sur Vercel
# Vérifiez dans le dashboard Vercel > Settings > Environment Variables
```

## Migration depuis les anciens liens Stripe

Si vous utilisez actuellement les anciens liens de paiement Stripe :
1. Créez les produits et Price IDs dans Stripe Dashboard
2. Ajoutez les Price IDs dans les variables d'environnement
3. Le système basculera automatiquement vers le SDK Stripe

Voir `STRIPE_PRICE_IDS_SETUP.md` pour plus de détails.

