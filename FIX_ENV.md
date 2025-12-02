# 🔧 Correction du fichier .env.local

## Problème identifié

Les variables `NEXT_PUBLIC_SUPABASE_ANON_KEY` et `SUPABASE_SERVICE_ROLE_KEY` sont **vides** ou **coupées sur plusieurs lignes**.

## ✅ Solution étape par étape

### 1. Ouvrir le fichier `.env.local`

Ouvrez le fichier `.env.local` dans votre éditeur (VS Code, etc.)

### 2. Trouver les lignes Supabase

Cherchez ces lignes :
```env
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
```

### 3. Ajouter les clés (UNE SEULE LIGNE)

**IMPORTANT :** Chaque clé doit être sur **UNE SEULE LIGNE**, sans retour à la ligne.

#### Pour `NEXT_PUBLIC_SUPABASE_ANON_KEY` :

1. Allez dans Supabase Dashboard > Settings > API Keys
2. Trouvez la clé **"anon public"**
3. Cliquez sur "Reveal" pour l'afficher
4. Copiez la clé **complète** (elle commence par `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`)
5. Collez-la **directement après le `=`**, sur la même ligne :

```env
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxwbnNmbGZ2amtvdnJpd3JrZ2xxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU2ODk2MDAsImV4cCI6MjA1MTI2NTYwMH0.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### Pour `SUPABASE_SERVICE_ROLE_KEY` :

1. Dans la même page Supabase, trouvez la clé **"service_role"**
2. Cliquez sur "Reveal" pour l'afficher
3. Copiez la clé **complète**
4. Collez-la **directement après le `=`**, sur la même ligne :

```env
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxwbnNmbGZ2amtvdnJpd3JrZ2xxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczNTY4OTYwMCwiZXhwIjoyMDUxMjY1NjAwfQ.yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
```

### 4. Vérifier le format

**✅ CORRECT :**
```env
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxwbnNmbGZ2amtvdnJpd3JrZ2xxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU2ODk2MDAsImV4cCI6MjA1MTI2NTYwMH0.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**❌ INCORRECT (coupé sur plusieurs lignes) :**
```env
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxwbnNmbGZ2amtvdnJpd3JrZ2xxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU2ODk2MDAsImV4cCI6MjA1MTI2NTYwMH0.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 5. Sauvegarder et redémarrer

1. **Sauvegardez** le fichier `.env.local`
2. **Arrêtez** complètement le serveur (Ctrl+C)
3. **Relancez** le serveur :
   ```bash
   npm run dev
   ```

## 🔍 Vérification

Après avoir ajouté les clés, vérifiez que tout est correct :

```bash
# Vérifier que les variables sont bien définies (sans afficher les valeurs)
cat .env.local | grep "NEXT_PUBLIC_SUPABASE_ANON_KEY" | grep -v "^#" | grep -v "^$"
```

Si la commande retourne une ligne avec une valeur (pas vide), c'est bon ✅

## ⚠️ Erreurs courantes

1. **Clé coupée sur plusieurs lignes** → Remettez-la sur une seule ligne
2. **Espaces avant/après le `=`** → Supprimez les espaces
3. **Guillemets autour de la clé** → Supprimez les guillemets
4. **Serveur non redémarré** → Redémarrez complètement le serveur

