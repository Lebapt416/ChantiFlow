-- Migration: Ajouter la colonne postal_code à la table sites
-- Date: 2025-01-XX
-- Description: Permet de stocker le code postal du chantier pour la météo et la localisation

-- Ajouter la colonne postal_code si elle n'existe pas déjà
DO $$ 
BEGIN
  IF NOT EXISTS (
    SELECT 1 
    FROM information_schema.columns 
    WHERE table_name = 'sites' 
    AND column_name = 'postal_code'
  ) THEN
    ALTER TABLE sites 
    ADD COLUMN postal_code VARCHAR(10);
    
    -- Ajouter un commentaire pour documenter la colonne
    COMMENT ON COLUMN sites.postal_code IS 'Code postal du chantier pour la météo et la localisation';
  END IF;
END $$;

-- Supprimer l'ancienne colonne address si elle existe (migration depuis address vers postal_code)
DO $$ 
BEGIN
  IF EXISTS (
    SELECT 1 
    FROM information_schema.columns 
    WHERE table_name = 'sites' 
    AND column_name = 'address'
  ) THEN
    ALTER TABLE sites DROP COLUMN address;
  END IF;
END $$;

