-- Migration : Ajouter un champ completed_at pour marquer les chantiers terminés
-- À exécuter dans l'éditeur SQL de Supabase

ALTER TABLE public.sites
  ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;

-- Ajouter un index pour accélérer les recherches de chantiers terminés
CREATE INDEX IF NOT EXISTS idx_sites_completed_at ON public.sites(completed_at);

