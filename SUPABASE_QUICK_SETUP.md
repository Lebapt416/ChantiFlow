# Configuration Supabase - Guide Rapide

## ⚠️ Erreur : "Supabase est mal configuré côté client"

Cette erreur signifie que les variables d'environnement Supabase ne sont pas définies.

## 🔧 Solution Rapide

### 1. Récupérer vos clés Supabase

1. Allez sur [app.supabase.com](https://app.supabase.com)
2. Sélectionnez votre projet
3. Allez dans **Settings** > **API**
4. Copiez les valeurs suivantes :
   - **Project URL** → `NEXT_PUBLIC_SUPABASE_URL`
   - **anon public** key → `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - **service_role** key → `SUPABASE_SERVICE_ROLE_KEY` (gardez cette clé secrète !)

### 2. Ajouter dans `.env.local`

Ouvrez le fichier `.env.local` à la racine du projet et ajoutez :

```env
NEXT_PUBLIC_SUPABASE_URL=https://votre-projet.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Exemple :**
```env
NEXT_PUBLIC_SUPABASE_URL=https://abcdefghijklmnop.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFiY2RlZmdoaWprbG1ub3AiLCJyb2xlIjoiYW5vbiIsImlhdCI6MTYzMDAwMDAwMCwiZXhwIjoxOTQ1NTYwMDAwfQ.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFiY2RlZmdoaWprbG1ub3AiLCJyb2xlIjoic2VydmljZV9yb2xlIiwiaWF0IjoxNjMwMDAwMDAwLCJleHAiOjE5NDU1NjAwMDB9.yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
```

### 3. Redémarrer le serveur de développement

Après avoir ajouté les variables, **redémarrez votre serveur** :

```bash
# Arrêtez le serveur (Ctrl+C)
# Puis relancez-le
npm run dev
```

## ✅ Vérification

Pour vérifier que les variables sont bien chargées :

1. Redémarrez le serveur de développement
2. L'erreur devrait disparaître
3. L'application devrait se charger correctement

## 🔒 Sécurité

- ✅ Le fichier `.env.local` est automatiquement ignoré par Git
- ⚠️ Ne commitez **JAMAIS** ces clés dans le code
- ⚠️ La clé `SUPABASE_SERVICE_ROLE_KEY` est très sensible, gardez-la secrète

## 📚 Documentation Complète

Pour plus de détails, consultez :
- `ENV_SETUP.md` : Guide complet des variables d'environnement
- `README.md` : Documentation générale du projet

## 🆘 Si le problème persiste

1. Vérifiez que le fichier `.env.local` est bien à la racine du projet
2. Vérifiez que les variables commencent bien par `NEXT_PUBLIC_` pour les clés côté client
3. Vérifiez qu'il n'y a pas d'espaces avant/après les valeurs
4. Redémarrez complètement le serveur de développement

