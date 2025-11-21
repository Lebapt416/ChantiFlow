# Dépannage - Erreur "Application error" sur /team

## Problème

Vous obtenez une erreur "Application error: a server-side exception has occurred" quand vous essayez d'ajouter quelqu'un à votre équipe.

## Solutions

### Solution 1 : Exécuter la migration SQL (RECOMMANDÉ)

**C'est la solution principale !** La migration SQL doit être exécutée pour que la fonctionnalité fonctionne.

1. **Allez sur Supabase** : https://app.supabase.com
2. **Sélectionnez votre projet**
3. **SQL Editor** → **New Query**
4. **Copiez-collez** le contenu complet de `migration-workers-to-account.sql`
5. **Cliquez sur "Run"** pour exécuter
6. **Vérifiez** qu'il n'y a pas d'erreur (devrait afficher "Success")
7. **Rechargez** la page `/team` et réessayez

### Solution 2 : Vérifier les logs Vercel

1. **Allez sur Vercel** : https://vercel.com
2. **Votre projet** → **Deployments** → Cliquez sur le dernier déploiement
3. **Logs** → Regardez les erreurs récentes
4. **Cherchez** les erreurs liées à :
   - `created_by` (colonne manquante)
   - `workers` (table)
   - `Resend` (service email)

### Solution 3 : Vérifier les variables d'environnement

Assurez-vous que ces variables sont configurées sur Vercel :

- ✅ `NEXT_PUBLIC_SUPABASE_URL`
- ✅ `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- ✅ `SUPABASE_SERVICE_ROLE_KEY`
- ✅ `NEXT_PUBLIC_APP_BASE_URL`
- ⚠️ `RESEND_API_KEY` (optionnel - si non configuré, les emails ne seront pas envoyés mais ça ne devrait pas bloquer)

### Solution 4 : Vérifier la structure de la table workers

Dans Supabase SQL Editor, exécutez :

```sql
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'workers'
AND table_schema = 'public';
```

Vous devriez voir :
- `id` (uuid)
- `site_id` (uuid, nullable)
- `name` (text)
- `email` (text, nullable)
- `role` (text, nullable)
- `created_at` (timestamp)
- **`created_by` (uuid, nullable)** ← Cette colonne doit exister !

Si `created_by` n'existe pas, exécutez la migration SQL.

### Solution 5 : Test rapide

Pour tester si la migration a été exécutée, dans Supabase SQL Editor :

```sql
SELECT * FROM workers LIMIT 1;
```

Si vous obtenez une erreur mentionnant `created_by`, la migration n'a pas été exécutée.

## Erreurs courantes

### "column workers.created_by does not exist"
→ **Solution** : Exécutez la migration SQL (`migration-workers-to-account.sql`)

### "RESEND_API_KEY non configuré"
→ **Solution** : C'est normal si vous n'avez pas configuré Resend. Les emails ne seront pas envoyés mais l'ajout de worker devrait fonctionner.

### "policy does not exist"
→ **Solution** : Exécutez la migration SQL (elle recrée les politiques)

## Après avoir exécuté la migration

1. **Rechargez** la page `/team`
2. **Essayez** d'ajouter un worker
3. **Vérifiez** que ça fonctionne

Si ça ne fonctionne toujours pas, vérifiez les logs Vercel pour voir l'erreur exacte.

