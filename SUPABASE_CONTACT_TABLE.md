# Création de la table contact_messages dans Supabase

## Problème
Les messages de contact ne s'affichent pas dans le dashboard administrateur car la table `contact_messages` n'existe pas encore dans Supabase.

## Solution : Créer la table

### Option 1 : Via l'éditeur SQL de Supabase (Recommandé)

1. Allez sur votre dashboard Supabase : https://supabase.com/dashboard
2. Sélectionnez votre projet
3. Cliquez sur **SQL Editor** dans le menu de gauche
4. Cliquez sur **New query**
5. Copiez-collez le SQL suivant :

```sql
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
```

6. Cliquez sur **Run** pour exécuter la requête
7. Vérifiez que la table a été créée en allant dans **Table Editor** → vous devriez voir `contact_messages`

### Option 2 : Via l'interface Table Editor

1. Allez dans **Table Editor**
2. Cliquez sur **New Table**
3. Nom de la table : `contact_messages`
4. Ajoutez les colonnes suivantes :
   - `id` : UUID, Primary Key, Default: `gen_random_uuid()`
   - `name` : TEXT, Not Null
   - `email` : TEXT, Not Null
   - `company` : TEXT, Nullable
   - `message` : TEXT, Not Null
   - `created_at` : TIMESTAMPTZ, Default: `now()`
5. Cliquez sur **Save**

## Vérification

Après avoir créé la table, testez en envoyant un nouveau message depuis le formulaire de contact. Le message devrait apparaître dans la section "Messages de Contact" du dashboard administrateur (`/analytics`).

## Permissions

La table utilise le client admin (`SUPABASE_SERVICE_ROLE_KEY`), donc les permissions sont automatiquement gérées. Aucune configuration RLS (Row Level Security) n'est nécessaire pour cette table.

