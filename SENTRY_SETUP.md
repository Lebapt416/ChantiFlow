# Configuration Sentry pour ChantiFlow

## Installation

Pour activer le monitoring d'erreurs avec Sentry, installez le package officiel :

```bash
npm install @sentry/nextjs
```

## Configuration

### 1. Initialiser Sentry dans Next.js

Exécutez la commande d'initialisation Sentry :

```bash
npx @sentry/nextjs wizard
```

Cette commande va :
- Créer `sentry.client.config.ts`
- Créer `sentry.server.config.ts`
- Créer `sentry.edge.config.ts`
- Modifier `next.config.js` pour intégrer Sentry
- Créer `instrumentation.ts`

### 2. Variables d'environnement

Ajoutez votre DSN Sentry dans `.env.local` :

```env
NEXT_PUBLIC_SENTRY_DSN=https://...@sentry.io/...
SENTRY_ORG=your-org
SENTRY_PROJECT=your-project
SENTRY_AUTH_TOKEN=your-auth-token
```

### 3. Configuration automatique

Une fois Sentry installé et configuré, le logger (`src/lib/logger.ts`) et le gestionnaire d'erreurs global (`src/app/global-error.tsx`) l'utiliseront automatiquement :

- ✅ Toutes les erreurs loggées avec `logger.error()` seront envoyées à Sentry
- ✅ Les erreurs globales non gérées seront capturées
- ✅ Le contexte (userId, siteId) sera automatiquement ajouté

## Fonctionnement

Le système est conçu pour fonctionner **avec ou sans Sentry** :

- **Avec Sentry** : Les erreurs sont automatiquement envoyées à Sentry avec le contexte complet
- **Sans Sentry** : Le système continue de fonctionner normalement, les erreurs sont juste loggées localement

## Avantages

1. **Monitoring en temps réel** : Toutes les erreurs sont trackées
2. **Contexte enrichi** : userId, siteId, et autres métadonnées sont automatiquement ajoutées
3. **Alertes** : Configurez des alertes dans Sentry pour être notifié des erreurs critiques
4. **Performance** : Suivez les performances de votre application

## Test

Pour tester l'intégration :

1. Installez Sentry
2. Configurez les variables d'environnement
3. Déclenchez une erreur (par exemple, dans une route API)
4. Vérifiez dans le dashboard Sentry que l'erreur apparaît

## Documentation

- [Documentation Sentry Next.js](https://docs.sentry.io/platforms/javascript/guides/nextjs/)
- [Configuration avancée](https://docs.sentry.io/platforms/javascript/guides/nextjs/manual-setup/)

