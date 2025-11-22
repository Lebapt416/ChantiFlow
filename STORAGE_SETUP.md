# Configuration du bucket Supabase Storage

## Problème
L'erreur "Bucket not found" apparaît lors de l'envoi de rapports avec photos.

## Solution

### Créer le bucket "reports" dans Supabase

1. Allez dans votre **Supabase Dashboard**
2. Cliquez sur **Storage** dans le menu de gauche
3. Cliquez sur **New bucket**
4. Configurez le bucket :
   - **Name**: `reports`
   - **Public bucket**: ✅ Activé (pour permettre l'accès aux photos)
   - **File size limit**: 5 MB (ou selon vos besoins)
   - **Allowed MIME types**: `image/*` (ou laissez vide pour tout accepter)
5. Cliquez sur **Create bucket**

### Configuration des politiques RLS (Row Level Security)

Pour permettre l'upload et la lecture des photos :

1. Allez dans **Storage** > **Policies** > **reports**
2. Ajoutez les politiques suivantes :

**Politique 1 : Upload autorisé pour les workers**
```sql
CREATE POLICY "Workers can upload reports"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'reports');
```

**Politique 2 : Lecture publique des photos**
```sql
CREATE POLICY "Public read access for reports"
ON storage.objects FOR SELECT
TO public
USING (bucket_id = 'reports');
```

### Alternative : Utiliser le service role key

Si vous utilisez le service role key (comme dans `createSupabaseAdminClient`), les politiques RLS sont contournées, mais le bucket doit quand même exister.

## Vérification

Après avoir créé le bucket, les rapports avec photos devraient fonctionner correctement. Si le bucket n'existe pas, le rapport sera envoyé sans photo (avec un avertissement dans les logs).

