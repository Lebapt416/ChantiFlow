# Instructions pour récupérer les clés Supabase

## ✅ Project URL déjà configuré

Votre Project URL a été ajouté : `https://lpnsflfvjkovriwrkglq.supabase.co`

## 🔑 Récupérer les clés API

### Étape 1 : Aller dans "API Keys"

1. Dans la sidebar de votre dashboard Supabase, cliquez sur **"API Keys"** (situé sous "Data API")
2. Vous verrez deux clés :

### Étape 2 : Copier les clés

#### 1. Clé "anon public" (clé publique)
- **Nom** : `anon` ou `anon public`
- **Usage** : Côté client (sécurisée, peut être exposée)
- **Variable** : `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- **Format** : Commence par `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

#### 2. Clé "service_role" (clé secrète)
- **Nom** : `service_role`
- **Usage** : Côté serveur uniquement (très sensible, ne jamais exposer)
- **Variable** : `SUPABASE_SERVICE_ROLE_KEY`
- **Format** : Commence par `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

### Étape 3 : Ajouter dans `.env.local`

Ouvrez le fichier `.env.local` et complétez avec vos clés :

```env
NEXT_PUBLIC_SUPABASE_URL=https://lpnsflfvjkovriwrkglq.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9... (votre clé anon public)
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9... (votre clé service_role)
```

### Étape 4 : Redémarrer le serveur

```bash
# Arrêtez le serveur (Ctrl+C)
npm run dev
```

## 🔒 Sécurité

- ✅ La clé `anon public` peut être exposée côté client (elle est limitée par les RLS)
- ⚠️ La clé `service_role` est très sensible, gardez-la secrète
- ⚠️ Ne commitez jamais ces clés dans Git (`.env.local` est déjà ignoré)

## 📍 Emplacement dans Supabase Dashboard

```
Settings
  └── PROJECT SETTINGS
      └── API Keys  ← Cliquez ici
```

Vous y verrez :
- **Project URL** : `https://lpnsflfvjkovriwrkglq.supabase.co` (déjà configuré ✅)
- **anon public** : À copier pour `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- **service_role** : À copier pour `SUPABASE_SERVICE_ROLE_KEY`

