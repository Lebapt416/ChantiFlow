-- Migration : Permettre aux workers d'être au niveau du compte (user)
-- À exécuter dans l'éditeur SQL de Supabase

-- 1. Rendre site_id nullable et email nullable (pour permettre des workers sans email)
ALTER TABLE public.workers 
  ALTER COLUMN site_id DROP NOT NULL,
  ALTER COLUMN email DROP NOT NULL;

-- 2. Ajouter created_by pour lier un worker à un compte utilisateur
ALTER TABLE public.workers 
  ADD COLUMN IF NOT EXISTS created_by UUID REFERENCES auth.users(id) ON DELETE CASCADE;

-- 3. Supprimer l'ancienne contrainte unique
ALTER TABLE public.workers 
  DROP CONSTRAINT IF EXISTS workers_site_id_email_key;

-- 4. Créer une nouvelle contrainte unique : 
--    - Si site_id est présent : unique par (site_id, email)
--    - Si site_id est NULL : unique par (created_by, email)
-- Note: On utilise une contrainte partielle pour gérer les deux cas
CREATE UNIQUE INDEX IF NOT EXISTS workers_site_email_unique 
  ON public.workers(site_id, email) 
  WHERE site_id IS NOT NULL AND email IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS workers_user_email_unique 
  ON public.workers(created_by, email) 
  WHERE site_id IS NULL AND created_by IS NOT NULL AND email IS NOT NULL;

-- 5. Mettre à jour les politiques RLS pour inclure les workers au niveau du compte
-- Supprimer les anciennes politiques si elles existent
DROP POLICY IF EXISTS "Users can view workers from their sites" ON public.workers;
DROP POLICY IF EXISTS "Users can view workers from their sites or account" ON public.workers;
DROP POLICY IF EXISTS "Users can create workers for their sites" ON public.workers;
DROP POLICY IF EXISTS "Users can create workers for their sites or account" ON public.workers;

-- Créer les nouvelles politiques
CREATE POLICY "Users can view workers from their sites or account"
  ON public.workers FOR SELECT
  USING (
    -- Workers liés à un chantier de l'utilisateur
    EXISTS (
      SELECT 1 FROM public.sites
      WHERE sites.id = workers.site_id
      AND sites.created_by = auth.uid()
    )
    OR
    -- Workers créés par l'utilisateur (au niveau du compte)
    workers.created_by = auth.uid()
  );

CREATE POLICY "Users can create workers for their sites or account"
  ON public.workers FOR INSERT
  WITH CHECK (
    -- Workers liés à un chantier de l'utilisateur
    (
      workers.site_id IS NOT NULL
      AND EXISTS (
        SELECT 1 FROM public.sites
        WHERE sites.id = workers.site_id
        AND sites.created_by = auth.uid()
      )
    )
    OR
    -- Workers créés par l'utilisateur (au niveau du compte)
    (
      workers.site_id IS NULL
      AND workers.created_by = auth.uid()
    )
  );

-- 6. Ajouter un index pour created_by
CREATE INDEX IF NOT EXISTS idx_workers_created_by ON public.workers(created_by);

