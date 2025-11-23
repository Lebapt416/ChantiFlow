-- Migration pour ajouter la colonne access_code à la table workers
-- Permet de stocker un code unique pour l'accès des workers à leur espace

-- Ajouter la colonne access_code si elle n'existe pas
DO $$ 
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'workers' AND column_name = 'access_code'
  ) THEN
    ALTER TABLE workers 
    ADD COLUMN access_code VARCHAR(8) UNIQUE;
    
    -- Créer un index pour améliorer les performances des requêtes
    CREATE INDEX IF NOT EXISTS idx_workers_access_code ON workers(access_code);
    
    -- Commentaire pour la documentation
    COMMENT ON COLUMN workers.access_code IS 'Code unique d''accès pour l''espace employé (8 caractères alphanumériques)';
  END IF;
END $$;
