-- ============================================================
-- Migration : Sécurisation des politiques RLS workers et reports
-- ChantiFlow — 2026-05-19
-- ============================================================
--
-- OBJECTIFS :
--   1. Remplacer la politique "Public read access for workers" (USING true)
--      par une politique à 3 branches : propriétaire authentifié,
--      worker de compte, ou accès QR via code d'accès valide.
--   2. Remplacer la politique "Public insert access for reports" (WITH CHECK true)
--      par une fonction SECURITY DEFINER qui valide la cohérence
--      task_id / worker_id / site_id avant tout INSERT.
--   3. Créer les index de performance associés.
-- ============================================================


-- ─────────────────────────────────────────────────────────────
-- SECTION 1 : INDEX DE PERFORMANCE
-- ─────────────────────────────────────────────────────────────

-- Lookup par code d'accès QR (branche 3 de la politique workers)
CREATE INDEX IF NOT EXISTS idx_workers_access_code
  ON public.workers(access_code)
  WHERE access_code IS NOT NULL;

-- Lookup workers par created_by (workers de compte sans site)
CREATE INDEX IF NOT EXISTS idx_workers_created_by
  ON public.workers(created_by)
  WHERE created_by IS NOT NULL;

-- Validation croisée rapide task_id + worker_id pour les rapports
CREATE INDEX IF NOT EXISTS idx_reports_task_worker
  ON public.reports(task_id, worker_id);


-- ─────────────────────────────────────────────────────────────
-- SECTION 2 : POLITIQUES RLS POUR LA TABLE workers
-- ─────────────────────────────────────────────────────────────

-- Supprimer les anciennes politiques qui seront remplacées
-- (idempotentes : IF EXISTS pour rejouer sans erreur)
DROP POLICY IF EXISTS "Public read access for workers"         ON public.workers;
DROP POLICY IF EXISTS "Users can view workers from their sites" ON public.workers;

-- ────────────────────────────────────────────────────────────
-- Nouvelle politique SELECT unifiée avec 3 branches :
--
--   Branche 1 – Manager authentifié propriétaire du site lié au worker
--   Branche 2 – Manager authentifié propriétaire du worker de compte (site_id NULL)
--   Branche 3 – Accès anonyme QR : le client anon doit appeler
--               SELECT set_config('app.worker_access_code', '<CODE>', true);
--               dans la même transaction avant son SELECT sur workers.
--               current_setting(..., true) renvoie NULL si non défini → pas d'accès.
-- ────────────────────────────────────────────────────────────
CREATE POLICY "Accès workers : propriétaire ou code QR valide"
  ON public.workers FOR SELECT
  USING (
    -- Branche 1 : utilisateur authentifié, propriétaire du chantier lié
    EXISTS (
      SELECT 1
        FROM public.sites s
       WHERE s.id = workers.site_id
         AND s.created_by = auth.uid()
    )

    OR

    -- Branche 2 : worker au niveau du compte (site_id NULL) appartenant à l'utilisateur
    (
      workers.site_id IS NULL
      AND workers.created_by = auth.uid()
    )

    OR

    -- Branche 3 : accès QR anonyme via code d'accès
    -- Sécurité : le code doit être non NULL, non vide, et correspondre exactement.
    -- current_setting retourne NULL si la variable de session n'est pas définie
    -- (paramètre missing_ok = true), ce qui rend la comparaison FALSE → pas d'accès.
    (
      workers.access_code IS NOT NULL
      AND workers.access_code <> ''
      AND workers.access_code = current_setting('app.worker_access_code', true)
    )
  );

COMMENT ON POLICY "Accès workers : propriétaire ou code QR valide" ON public.workers IS
  'SELECT autorisé si : (1) auth.uid() est propriétaire du site, '
  '(2) auth.uid() est créateur du worker de compte, '
  'ou (3) set_config(''app.worker_access_code'', code, true) fournit le bon code QR.';


-- ─────────────────────────────────────────────────────────────
-- SECTION 3 : FONCTION DE VALIDATION CROISÉE POUR reports
-- ─────────────────────────────────────────────────────────────

-- La fonction utilise SECURITY DEFINER pour bypasser le RLS lors de ses
-- propres SELECT internes (évite la récursion infinie) tout en étant appelée
-- dans le contexte de la politique RLS de la table reports.
--
-- Règles de validation :
--   1. La tâche (task_id) doit exister.
--   2. Le worker (worker_id) doit exister.
--   3. Si le worker a un site_id, il doit correspondre au site_id de la tâche.
--   4. Si le worker est au niveau du compte (site_id NULL), son created_by
--      doit correspondre au created_by du chantier de la tâche.

CREATE OR REPLACE FUNCTION public.fn_valider_rapport_coherent(
  p_task_id   UUID,
  p_worker_id UUID
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
-- Fixer le search_path évite les attaques par substitution de schéma
SET search_path = public
AS $$
DECLARE
  v_task_site_id    UUID;
  v_worker_site_id  UUID;
  v_worker_existe   BOOLEAN := FALSE;
BEGIN
  -- ── Étape 1 : la tâche existe-t-elle ? ──────────────────
  SELECT site_id
    INTO v_task_site_id
    FROM public.tasks
   WHERE id = p_task_id;

  -- FOUND est FALSE si aucune ligne retournée
  IF NOT FOUND OR v_task_site_id IS NULL THEN
    RETURN FALSE; -- Tâche introuvable ou sans site
  END IF;

  -- ── Étape 2 : le worker existe-t-il ? ───────────────────
  SELECT TRUE, site_id
    INTO v_worker_existe, v_worker_site_id
    FROM public.workers
   WHERE id = p_worker_id;

  IF NOT v_worker_existe THEN
    RETURN FALSE; -- Worker introuvable
  END IF;

  -- ── Étape 3 : vérification de cohérence ─────────────────
  IF v_worker_site_id IS NOT NULL THEN
    -- Worker directement lié à un chantier : les site_id doivent correspondre
    RETURN v_task_site_id = v_worker_site_id;
  ELSE
    -- Worker au niveau du compte (site_id NULL) :
    -- son créateur doit être le même que celui du chantier de la tâche
    RETURN EXISTS (
      SELECT 1
        FROM public.workers  w
        JOIN public.sites    s ON s.id = v_task_site_id
       WHERE w.id         = p_worker_id
         AND w.created_by = s.created_by
    );
  END IF;
END;
$$;

-- Révoquer l'accès direct depuis les rôles publics :
-- la fonction n'est appelable que via la politique RLS (SECURITY DEFINER)
REVOKE ALL ON FUNCTION public.fn_valider_rapport_coherent(UUID, UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.fn_valider_rapport_coherent(UUID, UUID)
  TO authenticated, anon, service_role;

COMMENT ON FUNCTION public.fn_valider_rapport_coherent(UUID, UUID) IS
  'Valide la cohérence d''un rapport avant INSERT : task_id et worker_id doivent '
  'appartenir au même chantier, directement ou via created_by. SECURITY DEFINER.';


-- ─────────────────────────────────────────────────────────────
-- SECTION 4 : POLITIQUES RLS POUR LA TABLE reports
-- ─────────────────────────────────────────────────────────────

-- Supprimer l'ancienne politique d'insertion aveugle
DROP POLICY IF EXISTS "Public insert access for reports" ON public.reports;

-- Nouvelle politique : INSERT validé par la fonction de cohérence métier
CREATE POLICY "Insertion rapport : cohérence tâche-worker-chantier"
  ON public.reports FOR INSERT
  WITH CHECK (
    public.fn_valider_rapport_coherent(task_id, worker_id)
  );

COMMENT ON POLICY "Insertion rapport : cohérence tâche-worker-chantier" ON public.reports IS
  'Autorise l''INSERT uniquement si fn_valider_rapport_coherent confirme que '
  'task_id et worker_id appartiennent au même chantier.';


-- ─────────────────────────────────────────────────────────────
-- SECTION 5 : VÉRIFICATION POST-MIGRATION
-- ─────────────────────────────────────────────────────────────

-- Lister les politiques actives sur workers et reports
SELECT
  tablename,
  policyname,
  cmd        AS operation,
  permissive AS type
FROM pg_policies
WHERE schemaname = 'public'
  AND tablename IN ('workers', 'reports')
ORDER BY tablename, cmd;

-- Vérifier les index créés
SELECT
  indexname,
  tablename,
  indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename IN ('workers', 'reports')
  AND indexname LIKE 'idx_%'
ORDER BY tablename, indexname;
