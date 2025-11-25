# Guide de Tests - ChantiFlow

Ce document décrit la stratégie de tests mise en place pour garantir la qualité du code.

## 🧪 Tests Unitaires (Vitest)

### Installation

Les dépendances sont déjà installées. Pour lancer les tests :

```bash
npm run test          # Exécute tous les tests une fois
npm run test:watch    # Mode watch (re-exécute à chaque changement)
npm run test:ui       # Interface graphique Vitest
```

### Structure

Les tests unitaires sont situés à côté des fichiers qu'ils testent avec l'extension `.test.ts` ou `.test.tsx`.

Exemple : `src/lib/ai/local-planning.test.ts` teste `src/lib/ai/local-planning.ts`

### Exemples de tests

- **Tests de logique métier** : Vérification des algorithmes de planning
- **Tests de règles** : Respect des contraintes (week-ends, heures de travail)
- **Tests de dépendances** : Ordre correct des tâches

## 🎭 Tests End-to-End (Playwright)

### Installation

Les navigateurs Playwright sont déjà installés. Pour lancer les tests E2E :

```bash
npm run test:e2e      # Exécute tous les tests E2E
npm run test:e2e:ui   # Interface graphique Playwright
```

### Structure

Les tests E2E sont dans le dossier `tests/` avec l'extension `.spec.ts`.

### Exemples de tests

- **Tests d'authentification** : Vérification du flux de connexion
- **Tests de navigation** : Vérification des parcours utilisateur
- **Tests de formulaires** : Validation et soumission

## 🔄 CI/CD (GitHub Actions)

Le workflow CI/CD s'exécute automatiquement sur chaque push et pull request.

### Jobs exécutés

1. **Lint & Type Check** : Vérification du code avec ESLint et TypeScript
2. **Unit Tests** : Exécution des tests unitaires avec Vitest
3. **Build** : Compilation de l'application Next.js
4. **E2E Tests** : Exécution des tests Playwright

### Configuration

Le workflow est défini dans `.github/workflows/ci.yml`.

## 📊 Coverage

Pour générer un rapport de couverture :

```bash
npm run test -- --coverage
```

Le rapport sera généré dans `coverage/`.

## 🐛 Debugging

### Tests unitaires

Utilisez `console.log()` ou les breakpoints de votre IDE.

### Tests E2E

- Utilisez `page.pause()` dans votre test pour ouvrir le mode debug
- Utilisez `npm run test:e2e:ui` pour l'interface graphique avec timeline

## 📝 Bonnes pratiques

1. **Tests unitaires** : Testez la logique métier, pas l'UI
2. **Tests E2E** : Testez les parcours utilisateur critiques
3. **Nommage** : Utilisez des noms descriptifs (`should avoid weekends in planning`)
4. **Isolation** : Chaque test doit être indépendant
5. **Mocking** : Utilisez des mocks pour les dépendances externes

## 🚀 Prochaines étapes

- [ ] Ajouter des tests pour les composants React
- [ ] Intégrer Sentry pour le tracking d'erreurs
- [ ] Ajouter des tests de performance
- [ ] Configurer le coverage minimum requis

