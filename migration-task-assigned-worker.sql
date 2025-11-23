-- Migration pour ajouter la colonne assigned_worker_id à la table tasks
-- Permet d'assigner un worker à une tâche

-- Ajouter la colonne assigned_worker_id si elle n'existe pas
DO $$ 
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'tasks' AND column_name = 'assigned_worker_id'
  ) THEN
    ALTER TABLE tasks 
    ADD COLUMN assigned_worker_id UUID REFERENCES workers(id) ON DELETE SET NULL;
    
    -- Créer un index pour améliorer les performances des requêtes
    CREATE INDEX IF NOT EXISTS idx_tasks_assigned_worker_id ON tasks(assigned_worker_id);
    
    -- Commentaire pour la documentation
    COMMENT ON COLUMN tasks.assigned_worker_id IS 'Worker assigné à cette tâche (peut être NULL si non assignée)';
  END IF;
END $$;

