# 🔧 Dépannage : Variables d'environnement non détectées

## Problème

Les clés Supabase sont visibles dans l'éditeur mais Next.js ne les détecte pas.

## Solutions à essayer

### 1. Vérifier que le fichier est sauvegardé

Dans VS Code ou votre éditeur :
- Vérifiez qu'il n'y a pas de point blanc (●) à côté du nom du fichier dans l'onglet
- Si oui, sauvegardez avec `Cmd+S` (Mac) ou `Ctrl+S` (Windows/Linux)

### 2. Vérifier le format de la clé

**✅ FORMAT CORRECT :**
```env
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxwbnNmbGZ2amtvdnJpd3JrZ2xxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU2ODk2MDAsImV4cCI6MjA1MTI2NTYwMH0.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**❌ FORMATS INCORRECTS :**
```env
# Avec guillemets
NEXT_PUBLIC_SUPABASE_ANON_KEY="eyJhbGci..."

# Avec espaces
NEXT_PUBLIC_SUPABASE_ANON_KEY = eyJhbGci...

# Coupé sur plusieurs lignes
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGci...
eyJpc3MiOiJzdXBhYmFzZSIs...
```

### 3. Redémarrer complètement le serveur

**IMPORTANT :** Next.js charge les variables d'environnement au démarrage. Un simple rechargement de page ne suffit pas.

```bash
# 1. Arrêtez complètement le serveur (Ctrl+C)
# 2. Attendez quelques secondes
# 3. Relancez
npm run dev
```

### 4. Vérifier que le fichier est au bon endroit

Le fichier `.env.local` doit être à la **racine du projet**, au même niveau que `package.json` :

```
ChantiFlow/
├── .env.local          ← ICI
├── package.json
├── next.config.ts
└── src/
```

### 5. Vérifier les caractères invisibles

Parfois, des caractères invisibles peuvent casser la lecture. Pour vérifier :

```bash
# Afficher la ligne avec les caractères invisibles
cat .env.local | grep "NEXT_PUBLIC_SUPABASE_ANON_KEY" | cat -v
```

Si vous voyez des caractères étranges, supprimez la ligne et réécrivez-la.

### 6. Vider le cache Next.js

```bash
# Supprimer le cache Next.js
rm -rf .next

# Redémarrer
npm run dev
```

### 7. Vérifier dans le terminal

Pour vérifier que Next.js charge bien les variables :

```bash
# Dans un nouveau terminal, vérifier
node -e "require('dotenv').config({ path: '.env.local' }); console.log('ANON_KEY:', process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ? '✅ Présente (' + process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY.length + ' caractères)' : '❌ Absente');"
```

## Checklist finale

- [ ] Fichier `.env.local` sauvegardé
- [ ] Clés sur une seule ligne (pas de retour à la ligne)
- [ ] Pas d'espaces avant/après le `=`
- [ ] Pas de guillemets autour des clés
- [ ] Fichier à la racine du projet
- [ ] Serveur complètement redémarré (pas juste rechargé)
- [ ] Cache `.next` vidé si nécessaire

## Si le problème persiste

1. Créez un nouveau fichier `.env.local` propre
2. Copiez-collez le contenu ligne par ligne
3. Vérifiez chaque ligne
4. Redémarrez le serveur

