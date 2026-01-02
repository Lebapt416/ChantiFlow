-- Créer la table contact_messages pour stocker les messages du formulaire de contact
CREATE TABLE IF NOT EXISTS contact_messages (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  company TEXT,
  message TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Créer un index sur created_at pour les requêtes de tri
CREATE INDEX IF NOT EXISTS idx_contact_messages_created_at ON contact_messages(created_at DESC);

-- Ajouter des commentaires pour la documentation
COMMENT ON TABLE contact_messages IS 'Messages envoyés depuis le formulaire de contact';
COMMENT ON COLUMN contact_messages.name IS 'Nom complet de la personne';
COMMENT ON COLUMN contact_messages.email IS 'Email de contact';
COMMENT ON COLUMN contact_messages.company IS 'Entreprise (optionnel)';
COMMENT ON COLUMN contact_messages.message IS 'Message du formulaire';
COMMENT ON COLUMN contact_messages.created_at IS 'Date de création du message';

