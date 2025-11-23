# Guide d'assignation de tâches aux workers

## Problème : "Je ne peux pas assigner mes tâches à des personnes de mon équipe"

### Étape 1 : Vérifier que la migration SQL est exécutée

La colonne `assigned_worker_id` doit exister dans la table `tasks`. 

**Exécuter la migration :**
1. Allez dans Supabase Dashboard → SQL Editor
2. Copiez-collez le contenu de `migration-task-assigned-worker.sql`
3. Exécutez la requête

**Vérifier que la colonne existe :**
```sql
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'tasks' AND column_name = 'assigned_worker_id';
```

Si la requête retourne un résultat, la colonne existe. Sinon, exécutez la migration.

### Étape 2 : Vérifier que vous avez des workers dans votre équipe

1. Allez sur la page "Équipe" (`/team`)
2. Vérifiez que vous avez des membres d'équipe créés
3. Les workers peuvent être :
   - Au niveau du compte (sans site_id) - réutilisables
   - Assignés à un chantier spécifique (avec site_id)

### Étape 3 : Vérifier les permissions

Les workers doivent avoir `created_by` égal à votre ID utilisateur pour être assignables.

### Étape 4 : Tester l'assignation

1. Allez sur la page "Tâches" (`/tasks`)
2. Pour chaque tâche en cours, vous devriez voir un bouton "Assigner"
3. Cliquez sur le bouton
4. Sélectionnez un worker dans la liste déroulante
5. Cliquez sur "Assigner"

### Dépannage

**Si le bouton "Assigner" n'apparaît pas :**
- Vérifiez que vous avez des tâches en cours
- Vérifiez que la migration SQL a été exécutée

**Si la liste des workers est vide :**
- Allez sur `/team` et ajoutez des membres
- Vérifiez que les workers ont `created_by` égal à votre ID

**Si vous voyez une erreur "assigned_worker_id n'existe pas" :**
- Exécutez la migration SQL `migration-task-assigned-worker.sql`

**Si vous voyez "Aucun membre d'équipe disponible" :**
- Allez sur `/team` et créez des membres d'équipe
- Les workers doivent être créés par vous (created_by = votre ID)

### Structure de la base de données

La table `tasks` doit avoir :
- `id` (UUID)
- `site_id` (UUID)
- `title` (text)
- `status` (text)
- `assigned_worker_id` (UUID, nullable) ← **Cette colonne doit exister**

La table `workers` doit avoir :
- `id` (UUID)
- `name` (text)
- `email` (text, nullable)
- `role` (text, nullable)
- `created_by` (UUID) ← **Doit être votre ID utilisateur**
- `site_id` (UUID, nullable)

