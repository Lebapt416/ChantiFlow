-- Migration : Ajouter un code d'accès unique pour chaque worker
-- À exécuter dans l'éditeur SQL de Supabase

-- 1. Ajouter la colonne access_code
ALTER TABLE public.workers 
  ADD COLUMN IF NOT EXISTS access_code VARCHAR(8) UNIQUE;

-- 2. Créer un index pour les recherches rapides
CREATE INDEX IF NOT EXISTS idx_workers_access_code ON public.workers(access_code);

-- 3. Générer des codes pour les workers existants (optionnel)
-- UPDATE public.workers 
-- SET access_code = UPPER(SUBSTRING(MD5(RANDOM()::TEXT || id::TEXT) FROM 1 FOR 8))
-- WHERE access_code IS NULL;

