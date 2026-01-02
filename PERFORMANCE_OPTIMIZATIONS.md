# Optimisations de Performance - ChantiFlow

## 🎯 Problèmes identifiés et solutions appliquées

### 1. ✅ Streaming & Loading UI (FCP critique)

**Problème** : FCP de 26.9s dû à l'absence de loading states et au blocage sur les requêtes serveur.

**Solutions appliquées** :
- ✅ Création de `src/app/loading.tsx` avec skeleton pour la landing page
- ✅ Création de `src/app/home/loading.tsx` avec skeleton pour la page d'accueil
- ✅ Utilisation de `<Suspense>` pour la section Pricing (ne bloque plus le FCP)
- ✅ HeroSection rendu statique initialement, animations chargées après montage

**Impact attendu** : Réduction du FCP de ~27s à <2s

### 2. ✅ Optimisation des requêtes (Caching & Parallélisation)

**Problème** : Requêtes Supabase en waterfall et absence de caching.

**Solutions appliquées** :
- ✅ Parallélisation des requêtes dans `/home` avec `Promise.all`
- ✅ Vérification d'authentification déplacée dans Suspense (non-bloquant)
- ⚠️ **À faire** : Ajouter `revalidate` sur les routes statiques

**Impact attendu** : Réduction du TTFB de ~20s à <1s

### 3. ✅ Optimisation LCP (Images)

**Problème** : Image hero sans optimisation complète.

**Solutions appliquées** :
- ✅ Image hero avec `priority` (déjà présent)
- ✅ Ajout de `quality={85}` pour réduire la taille
- ✅ `sizes` correctement configuré

**Impact attendu** : LCP réduit de 8.8s à <2.5s

### 4. ✅ Optimisation des imports lourds

**Problème** : `framer-motion` chargé immédiatement, bloquant le rendu.

**Solutions appliquées** :
- ✅ HeroSection rendu statique initialement
- ✅ Animations chargées après montage avec `useEffect`
- ✅ Fallback statique pour le FCP

**Impact attendu** : Réduction du temps de parsing JS

## 📋 Actions supplémentaires recommandées

### 1. Ajouter du caching Next.js

Pour les pages qui changent peu souvent, ajouter `revalidate` :

```typescript
// Dans src/app/page.tsx
export const revalidate = 3600; // Cache 1 heure

// Pour les données utilisateur, utiliser un cache plus court
export const revalidate = 60; // Cache 1 minute
```

### 2. Optimiser les fonts

Les fonts Google sont déjà optimisées avec `next/font`, mais vérifier :
- ✅ `display: 'swap'` (déjà géré par Next.js)
- ✅ Preload des fonts critiques

### 3. Lazy load des composants lourds

Pour les sections en bas de page :
```typescript
const FeaturesSection = lazy(() => import('@/components/landing/features-section'));
const FaqSection = lazy(() => import('@/components/landing/faq-section'));
```

### 4. Optimiser Supabase queries

Ajouter des index sur les colonnes fréquemment queryées :
- `sites.created_by`
- `tasks.site_id`
- `workers.site_id`

### 5. Vérifier les métriques après déploiement

Après déploiement sur Vercel, vérifier :
- FCP devrait être < 2s
- LCP devrait être < 2.5s
- TTFB devrait être < 1s

## 🔍 Monitoring

Utiliser Vercel Speed Insights pour suivre :
- FCP (First Contentful Paint)
- LCP (Largest Contentful Paint)
- CLS (Cumulative Layout Shift)
- TTFB (Time to First Byte)

## 📝 Notes techniques

- Les `loading.tsx` sont automatiquement utilisés par Next.js App Router
- Le Suspense permet le streaming des composants asynchrones
- Les images avec `priority` sont préchargées
- Le rendu statique initial évite le "flash of unstyled content"

