# Plan d'Exécution Technique - Transformation Production-Grade ChantiFlow

**Objectif**: Transformer ChantiFlow en SaaS "indestructible" pour 1000 clients et 10M€ de valorisation.

**Date**: $(date)
**CTO Expert**: Analyse et implémentation

---

## 📊 ANALYSE DE L'EXISTANT

### ✅ Points Forts Identifiés
- **Offline Sync**: Hook `useOfflineSync.ts` fonctionnel avec IndexedDB
- **IA Planning**: `improved-planning.ts` avec intégration météo via `weather-cache.ts`
- **Dashboard Analytics**: Recharts déjà intégré, graphiques basiques présents
- **Tests**: Playwright configuré, 1 test auth existant
- **Logger**: Système de logging structuré prêt pour Sentry

### ⚠️ Points d'Amélioration Critiques

#### 1. MODE HORS-LIGNE (useOfflineSync.ts)
**Problèmes actuels**:
- ❌ Pas de gestion de conflits (timestamps)
- ❌ Pas de file d'attente robuste (Queue)
- ❌ Compteur de rapports pas toujours précis
- ❌ Pas de versioning des données

**Solutions à implémenter**:
- ✅ Ajouter `lastModified` timestamp à chaque `PendingReport`
- ✅ Implémenter stratégie "Last Write Wins" avec comparaison de timestamps
- ✅ Créer système de Queue avec priorités (FIFO + retry logic)
- ✅ Améliorer compteur avec IndexedDB index sur `createdAt`

#### 2. INTELLIGENCE ARTIFICIELLE (improved-planning.ts)
**Problèmes actuels**:
- ❌ Prompts Gemini génériques, pas de contraintes ouvriers
- ❌ Pas d'intégration météo dans les prompts
- ❌ Pas de fonction `autoAdjustPlanning` pour risques

**Solutions à implémenter**:
- ✅ Enrichir prompts avec compétences ouvriers disponibles
- ✅ Intégrer données météo réelles dans les prompts Gemini
- ✅ Créer `autoAdjustPlanning()` qui utilise `risk-analysis.ts`
- ✅ Ajouter logique de décalage automatique des tâches

#### 3. UX DASHBOARD & CONVERSION
**Problèmes actuels**:
- ❌ Graphiques financiers manquants (MRR, rétention)
- ❌ `report-form.tsx` pas optimisé terrain (petits boutons)
- ❌ Pas de dictée vocale
- ❌ Photos uploadées sans compression

**Solutions à implémenter**:
- ✅ Graphique MRR mensuel avec projection
- ✅ Graphique taux de rétention (cohortes)
- ✅ Boutons 44px minimum (accessibilité terrain)
- ✅ Web Speech API pour description vocale
- ✅ Compression images avant upload (canvas + toBlob)

#### 4. FIABILITÉ & MONITORING
**Problèmes actuels**:
- ❌ Sentry mentionné mais pas intégré
- ❌ Pas de tracking sur Server Actions critiques
- ❌ Test Playwright incomplet (pas de scénario offline)

**Solutions à implémenter**:
- ✅ Installer `@sentry/nextjs`
- ✅ Wrapper Sentry pour toutes Server Actions (Auth, Stripe, Sync)
- ✅ Test Playwright: Scan QR → Saisie offline → Réseau → Synchro

---

## 🎯 PLAN D'EXÉCUTION DÉTAILLÉ

### PHASE 1: SÉCURISATION MODE HORS-LIGNE (Priorité: CRITIQUE)

#### 1.1 Amélioration useOfflineSync.ts
**Fichier**: `src/hooks/useOfflineSync.ts`

**Modifications**:
```typescript
// Ajouter au type PendingReport:
- lastModified: number; // Timestamp de dernière modification
- version: number; // Version pour résolution de conflits
- priority: 'high' | 'medium' | 'low'; // Priorité dans la queue

// Nouvelle fonction:
- resolveConflict(local: PendingReport, server: PendingReport): PendingReport
  // Stratégie: Last Write Wins basé sur lastModified

// Amélioration IndexedDB:
- Index sur 'lastModified' pour tri rapide
- Index sur 'priority' pour queue prioritaire
- Index sur 'retryCount' pour gestion des échecs
```

**Migration IndexedDB**:
- Version 2: Ajouter colonnes `lastModified`, `version`, `priority`
- Migration automatique lors de l'upgrade

#### 1.2 File d'Attente (Queue) Robuste
**Nouveau fichier**: `src/lib/offline-queue.ts`

**Fonctionnalités**:
- Queue FIFO avec support priorités
- Retry logic avec backoff exponentiel
- Gestion des échecs persistants (max 5 retries)
- Logging des tentatives pour debugging

#### 1.3 Amélioration offline-indicator.tsx
**Fichier**: `src/components/offline-indicator.tsx`

**Modifications**:
- Afficher nombre exact: `{pendingCount} rapport(s) en attente`
- Badge avec couleur selon priorité (high = rouge)
- Animation lors de la synchro
- Tooltip avec détails (IDs des rapports en attente)

---

### PHASE 2: INTELLIGENCE ARTIFICIELLE & AUTOMATISATION

#### 2.1 Optimisation Prompts Gemini
**Fichier**: `src/lib/ai/improved-planning.ts`

**Modifications**:
```typescript
// Nouvelle fonction enrichie:
async function generatePlanningWithConstraints(
  tasks: Task[],
  workers: Worker[],
  deadline: string | null,
  location?: string,
  weatherData?: WeatherData
): Promise<PlanningResult>

// Prompt enrichi avec:
- Liste des compétences ouvriers disponibles
- Contraintes météo (pluie, vent, température)
- Historique de performance des ouvriers
- Règles de sécurité (ex: pas de toiture par grand vent)
```

**Intégration weather-cache.ts**:
- Utiliser données météo réelles dans le prompt
- Exemple: "Éviter travaux extérieurs si pluie > 5mm prévue"

#### 2.2 Fonction autoAdjustPlanning
**Nouveau fichier**: `src/lib/ai/auto-adjust-planning.ts`

**Fonctionnalités**:
```typescript
export async function autoAdjustPlanning(
  siteId: string,
  riskLevel: 'high' | 'medium' | 'low',
  riskReason: string
): Promise<{
  adjusted: boolean;
  newDates: Array<{ taskId: string; newStartDate: string; newEndDate: string }>;
  warnings: string[];
}>

// Logique:
1. Récupérer planning actuel
2. Analyser risque via risk-analysis.ts
3. Si risque HIGH: Décale toutes tâches critiques de +2 jours
4. Si risque MEDIUM: Décale tâches extérieures de +1 jour
5. Si risque LOW: Avertissement seulement
6. Mettre à jour dates dans Supabase
7. Envoyer notification au chef de chantier
```

**Intégration risk-analysis.ts**:
- Appeler `analyserRisqueRetard()` avant ajustement
- Utiliser `risque_pourcentage` pour déterminer niveau

---

### PHASE 3: UX DASHBOARD & CONVERSION

#### 3.1 Graphiques Financiers
**Fichier**: `src/components/analytics-dashboard.tsx`

**Nouveaux graphiques**:
1. **MRR Mensuel** (LineChart):
   - Courbe MRR sur 12 derniers mois
   - Projection 3 mois (tendance)
   - Breakdown par plan (Basic, Pro, Plus)

2. **Taux de Rétention** (BarChart):
   - Cohorte mensuelle (ex: Jan 2024)
   - % utilisateurs actifs après 1, 3, 6, 12 mois
   - Comparaison avec benchmark industrie (70% = bon)

3. **Churn Rate** (PieChart):
   - % utilisateurs perdus par mois
   - Raisons principales (si disponibles)

**Données nécessaires**:
- Requête Supabase: `subscriptions` avec `created_at`, `canceled_at`, `plan`
- Calcul MRR: `SUM(monthly_price WHERE status = 'active')`
- Calcul rétention: Cohorte analysis sur `users.created_at`

#### 3.2 Optimisation report-form.tsx
**Fichier**: `src/app/qr/[siteId]/report-form.tsx`

**Modifications**:

1. **Boutons plus larges** (44px minimum):
```tsx
<button className="min-h-[44px] min-w-[120px] ...">
  // Accessibilité terrain (gants, écran sale)
</button>
```

2. **Dictée vocale** (Web Speech API):
```typescript
// Nouveau hook: useSpeechRecognition()
const { isListening, transcript, startListening, stopListening } = useSpeechRecognition();

// Bouton micro:
<button onClick={isListening ? stopListening : startListening}>
  {isListening ? '🛑 Arrêter' : '🎤 Dictée'}
</button>

// Auto-remplir description avec transcript
```

3. **Compression photos**:
```typescript
// Nouvelle fonction: compressImage(file: File, maxWidth: 1920, quality: 0.8)
async function compressImage(file: File): Promise<File> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d')!;
        
        // Calculer dimensions (max 1920px)
        const maxWidth = 1920;
        const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
        canvas.width = img.width * ratio;
        canvas.height = img.height * ratio;
        
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob(
          (blob) => {
            const compressedFile = new File([blob!], file.name, { type: 'image/jpeg' });
            resolve(compressedFile);
          },
          'image/jpeg',
          0.8 // 80% quality
        );
      };
      img.src = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  });
}

// Utilisation avant upload:
const compressedPhoto = await compressImage(photoFile);
formData.append('photo', compressedPhoto);
```

---

### PHASE 4: FIABILITÉ & MONITORING

#### 4.1 Intégration Sentry
**Installation**:
```bash
npm install @sentry/nextjs
npx @sentry/nextjs wizard
```

**Fichiers à créer**:
- `sentry.client.config.ts`
- `sentry.server.config.ts`
- `sentry.edge.config.ts`

**Wrapper Server Actions**:
**Nouveau fichier**: `src/lib/sentry-wrapper.ts`
```typescript
import * as Sentry from '@sentry/nextjs';

export function withSentryTracking<T extends (...args: any[]) => Promise<any>>(
  actionName: string,
  action: T
): T {
  return (async (...args: Parameters<T>) => {
    const transaction = Sentry.startTransaction({
      name: `ServerAction:${actionName}`,
      op: 'server.action',
    });

    try {
      const result = await action(...args);
      transaction.setStatus('ok');
      return result;
    } catch (error) {
      transaction.setStatus('internal_error');
      Sentry.captureException(error, {
        tags: { action: actionName },
        extra: { args },
      });
      throw error;
    } finally {
      transaction.finish();
    }
  }) as T;
}
```

**Actions critiques à wrapper**:
- `src/app/login/actions.ts`: `loginAction`, `signupAction`
- `src/app/qr/[siteId]/actions.ts`: `submitReportAction`
- `src/app/dashboard/actions.ts`: `createSiteAction`
- `src/app/team/actions/route.ts`: `POST` (approve/reject)
- Toutes les actions Stripe (si présentes)

#### 4.2 Test Playwright Complet
**Nouveau fichier**: `tests/offline-sync.spec.ts`

**Scénario**:
```typescript
test('Scan QR → Saisie Hors-ligne → Réseau → Synchro', async ({ page, context }) => {
  // 1. Se connecter comme ouvrier
  await page.goto('/worker/login');
  await page.fill('[name="access_code"]', 'TEST1234');
  await page.click('button[type="submit"]');
  
  // 2. Scanner QR code (simuler)
  await page.goto('/qr/[siteId]');
  // Vérifier que le formulaire est chargé
  
  // 3. Couper le réseau (offline)
  await context.setOffline(true);
  
  // 4. Remplir et soumettre rapport
  await page.fill('[name="description"]', 'Test offline');
  await page.setInputFiles('[name="photo"]', 'test-image.jpg');
  await page.click('button[type="submit"]');
  
  // 5. Vérifier message "sauvegardé localement"
  await expect(page.locator('text=sauvegardé localement')).toBeVisible();
  
  // 6. Rétablir réseau
  await context.setOffline(false);
  
  // 7. Attendre synchro automatique (max 10s)
  await page.waitForSelector('text=synchronisé', { timeout: 10000 });
  
  // 8. Vérifier que le rapport apparaît dans la liste
  await page.goto('/reports');
  await expect(page.locator('text=Test offline')).toBeVisible();
});
```

---

## 📋 ORDRE D'EXÉCUTION RECOMMANDÉ

1. **PHASE 1** (Critique): Mode hors-ligne → Impact immédiat sur fiabilité
2. **PHASE 4** (Critique): Sentry → Monitoring avant scaling
3. **PHASE 2** (Important): IA → Différenciation produit
4. **PHASE 3** (Important): UX → Conversion et rétention

---

## 🔒 CONSIGNES DE QUALITÉ

### TypeScript Strict
- ✅ `tsconfig.json`: `"strict": true`
- ✅ 0 `any` explicites (utiliser `unknown` si nécessaire)
- ✅ Types pour tous les retours de fonctions

### RLS Supabase
- ✅ Vérifier toutes les nouvelles tables
- ✅ Policies pour `SELECT`, `INSERT`, `UPDATE`, `DELETE`
- ✅ Tests des policies dans les migrations

### Documentation
- ✅ JSDoc pour toutes les fonctions publiques
- ✅ README.md mis à jour
- ✅ CHANGELOG.md pour chaque phase

---

## 📊 MÉTRIQUES DE SUCCÈS

- **Mode Hors-ligne**: 0% perte de données, synchro < 5s après reconnexion
- **IA**: 90%+ précision dans les ajustements de planning
- **UX**: Temps de saisie rapport réduit de 40% (dictée + compression)
- **Monitoring**: 100% Server Actions critiques trackées dans Sentry
- **Tests**: 80%+ coverage E2E (scénarios critiques)

---

**Prochaine étape**: Validation du plan → Implémentation Phase 1

