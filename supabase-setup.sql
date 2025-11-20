-- ============================================
-- Script de configuration Supabase pour ChantiFlow
-- ============================================
-- À exécuter dans l'éditeur SQL de Supabase
-- https://app.supabase.com → SQL Editor

-- 1. Créer les tables (si pas déjà fait)
-- ============================================

-- Table des chantiers
CREATE TABLE IF NOT EXISTS public.sites (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  deadline DATE,
  created_by UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table des tâches
CREATE TABLE IF NOT EXISTS public.tasks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  site_id UUID NOT NULL REFERENCES public.sites(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  required_role TEXT,
  duration_hours INTEGER,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'done')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table des employés
CREATE TABLE IF NOT EXISTS public.workers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  site_id UUID NOT NULL REFERENCES public.sites(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  role TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(site_id, email)
);

-- Table des rapports
CREATE TABLE IF NOT EXISTS public.reports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id UUID NOT NULL REFERENCES public.tasks(id) ON DELETE CASCADE,
  worker_id UUID NOT NULL REFERENCES public.workers(id) ON DELETE CASCADE,
  photo_url TEXT,
  description TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Activer Row Level Security (RLS)
-- ============================================

ALTER TABLE public.sites ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.workers ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;

-- 3. Politiques RLS pour sites
-- ============================================

-- Les utilisateurs peuvent voir leurs propres chantiers
CREATE POLICY "Users can view their own sites"
  ON public.sites FOR SELECT
  USING (auth.uid() = created_by);

-- Les utilisateurs peuvent créer leurs propres chantiers
CREATE POLICY "Users can create their own sites"
  ON public.sites FOR INSERT
  WITH CHECK (auth.uid() = created_by);

-- Les utilisateurs peuvent modifier leurs propres chantiers
CREATE POLICY "Users can update their own sites"
  ON public.sites FOR UPDATE
  USING (auth.uid() = created_by);

-- Les utilisateurs peuvent supprimer leurs propres chantiers
CREATE POLICY "Users can delete their own sites"
  ON public.sites FOR DELETE
  USING (auth.uid() = created_by);

-- 4. Politiques RLS pour tasks
-- ============================================

-- Les utilisateurs peuvent voir les tâches de leurs chantiers
CREATE POLICY "Users can view tasks from their sites"
  ON public.tasks FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.sites
      WHERE sites.id = tasks.site_id
      AND sites.created_by = auth.uid()
    )
  );

-- Les utilisateurs peuvent créer des tâches pour leurs chantiers
CREATE POLICY "Users can create tasks for their sites"
  ON public.tasks FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.sites
      WHERE sites.id = tasks.site_id
      AND sites.created_by = auth.uid()
    )
  );

-- Les utilisateurs peuvent modifier les tâches de leurs chantiers
CREATE POLICY "Users can update tasks from their sites"
  ON public.tasks FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM public.sites
      WHERE sites.id = tasks.site_id
      AND sites.created_by = auth.uid()
    )
  );

-- 5. Politiques RLS pour workers
-- ============================================

-- Les utilisateurs peuvent voir les employés de leurs chantiers
CREATE POLICY "Users can view workers from their sites"
  ON public.workers FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.sites
      WHERE sites.id = workers.site_id
      AND sites.created_by = auth.uid()
    )
  );

-- Les utilisateurs peuvent créer des employés pour leurs chantiers
CREATE POLICY "Users can create workers for their sites"
  ON public.workers FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.sites
      WHERE sites.id = workers.site_id
      AND sites.created_by = auth.uid()
    )
  );

-- Accès public en lecture pour les employés (via QR)
CREATE POLICY "Public read access for workers"
  ON public.workers FOR SELECT
  USING (true);

-- 6. Politiques RLS pour reports
-- ============================================

-- Les utilisateurs peuvent voir les rapports de leurs chantiers
CREATE POLICY "Users can view reports from their sites"
  ON public.reports FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.tasks
      JOIN public.sites ON sites.id = tasks.site_id
      WHERE tasks.id = reports.task_id
      AND sites.created_by = auth.uid()
    )
  );

-- Accès public en écriture pour les rapports (via QR)
CREATE POLICY "Public insert access for reports"
  ON public.reports FOR INSERT
  WITH CHECK (true);

-- 7. Index pour améliorer les performances
-- ============================================

CREATE INDEX IF NOT EXISTS idx_tasks_site_id ON public.tasks(site_id);
CREATE INDEX IF NOT EXISTS idx_workers_site_id ON public.workers(site_id);
CREATE INDEX IF NOT EXISTS idx_workers_email ON public.workers(email);
CREATE INDEX IF NOT EXISTS idx_reports_task_id ON public.reports(task_id);
CREATE INDEX IF NOT EXISTS idx_reports_worker_id ON public.reports(worker_id);
CREATE INDEX IF NOT EXISTS idx_sites_created_by ON public.sites(created_by);

-- 8. Créer le bucket Storage pour les photos
-- ============================================
-- ⚠️ À faire manuellement dans l'interface Supabase :
-- Storage → Create bucket → Nom: "reports" → Public: OUI

-- 9. Vérifier la configuration
-- ============================================

-- Vérifier que RLS est activé
SELECT tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public' 
AND tablename IN ('sites', 'tasks', 'workers', 'reports');

-- Compter les politiques créées
SELECT schemaname, tablename, policyname 
FROM pg_policies 
WHERE schemaname = 'public';

