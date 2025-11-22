-- Migration pour ajouter le champ status à la table workers
-- Status peut être: 'pending' (en attente de validation), 'approved' (approuvé), 'rejected' (refusé)
-- Par défaut, les workers créés manuellement sont 'approved'

ALTER TABLE public.workers
  ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'approved';

-- Mettre à jour les workers existants pour qu'ils soient approuvés
UPDATE public.workers
  SET status = 'approved'
  WHERE status IS NULL;

-- Créer un index pour améliorer les performances des requêtes
CREATE INDEX IF NOT EXISTS idx_workers_status ON public.workers(status);
CREATE INDEX IF NOT EXISTS idx_workers_status_created_by ON public.workers(status, created_by);

