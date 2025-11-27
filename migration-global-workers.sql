-- Migration: rendre les workers globaux (déconnexion du site unique)
-- 1. site_id devient nullable
-- 2. ajout d'un access_token unique (clé de connexion persistante)
-- 3. ajout de la date du dernier login

DO $$
BEGIN
  -- site_id nullable
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'workers' AND column_name = 'site_id'
  ) THEN
    ALTER TABLE workers
      ALTER COLUMN site_id DROP NOT NULL;
  END IF;

  -- access_token
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'workers' AND column_name = 'access_token'
  ) THEN
    ALTER TABLE workers
      ADD COLUMN access_token UUID NOT NULL UNIQUE DEFAULT gen_random_uuid();
  END IF;

  -- last_login
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'workers' AND column_name = 'last_login'
  ) THEN
    ALTER TABLE workers
      ADD COLUMN last_login TIMESTAMP WITH TIME ZONE;
  END IF;
END $$;

