-- Vérification complète de la migration workers-to-account

-- 1. Vérifier que la colonne created_by existe
SELECT 
  column_name, 
  data_type, 
  is_nullable
FROM information_schema.columns
WHERE table_name = 'workers' 
  AND column_name = 'created_by';

-- 2. Vérifier que site_id est nullable
SELECT 
  column_name, 
  data_type, 
  is_nullable
FROM information_schema.columns
WHERE table_name = 'workers' 
  AND column_name = 'site_id';

-- 3. Vérifier que email est nullable
SELECT 
  column_name, 
  data_type, 
  is_nullable
FROM information_schema.columns
WHERE table_name = 'workers' 
  AND column_name = 'email';

-- 4. Vérifier toutes les politiques RLS
SELECT 
  policyname,
  cmd,
  permissive,
  roles
FROM pg_policies
WHERE tablename = 'workers'
ORDER BY policyname;

-- 5. Vérifier les index uniques
SELECT 
  indexname,
  indexdef
FROM pg_indexes
WHERE tablename = 'workers'
  AND indexdef LIKE '%UNIQUE%'
ORDER BY indexname;

-- 6. Vérifier l'index created_by
SELECT 
  indexname,
  indexdef
FROM pg_indexes
WHERE tablename = 'workers'
  AND indexname = 'idx_workers_created_by';

