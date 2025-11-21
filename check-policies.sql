-- Vérifier les politiques RLS pour la table workers
SELECT 
  schemaname,
  tablename,
  policyname,
  permissive,
  roles,
  cmd,
  qual,
  with_check
FROM pg_policies
WHERE tablename = 'workers'
ORDER BY policyname;

