-- Migration pour créer la table d'historique des plannings IA
-- Cette table permet l'entraînement continu de l'IA

CREATE TABLE IF NOT EXISTS ai_planning_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  site_id UUID NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
  site_name TEXT NOT NULL,
  deadline DATE,
  tasks_data JSONB NOT NULL, -- Données des tâches au moment de la génération
  workers_data JSONB NOT NULL, -- Données des workers au moment de la génération
  planning_result JSONB NOT NULL, -- Résultat du planning généré
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index pour améliorer les performances des requêtes
CREATE INDEX IF NOT EXISTS idx_ai_planning_history_site_id ON ai_planning_history(site_id);
CREATE INDEX IF NOT EXISTS idx_ai_planning_history_created_at ON ai_planning_history(created_at DESC);

-- Index GIN pour les recherches dans les données JSONB
CREATE INDEX IF NOT EXISTS idx_ai_planning_history_tasks_data ON ai_planning_history USING GIN(tasks_data);
CREATE INDEX IF NOT EXISTS idx_ai_planning_history_planning_result ON ai_planning_history USING GIN(planning_result);

-- Fonction pour mettre à jour updated_at automatiquement
CREATE OR REPLACE FUNCTION update_ai_planning_history_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger pour updated_at
DROP TRIGGER IF EXISTS trigger_update_ai_planning_history_updated_at ON ai_planning_history;
CREATE TRIGGER trigger_update_ai_planning_history_updated_at
  BEFORE UPDATE ON ai_planning_history
  FOR EACH ROW
  EXECUTE FUNCTION update_ai_planning_history_updated_at();

-- Commentaires pour la documentation
COMMENT ON TABLE ai_planning_history IS 'Historique des plannings générés par l''IA pour l''entraînement continu';
COMMENT ON COLUMN ai_planning_history.tasks_data IS 'Données JSON des tâches au moment de la génération du planning';
COMMENT ON COLUMN ai_planning_history.workers_data IS 'Données JSON des workers au moment de la génération du planning';
COMMENT ON COLUMN ai_planning_history.planning_result IS 'Résultat complet du planning généré par l''IA';

