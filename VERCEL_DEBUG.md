# 🔍 Diagnostic Problème Rechargement Infini - Vercel

## Problème
Le site se recharge automatiquement toutes les secondes sur Vercel.

## Solutions à tester

### 1. Vider le cache Vercel
Dans le dashboard Vercel :
- Allez dans **Settings** → **General**
- Cliquez sur **Clear Build Cache**
- Redéployez le projet

### 2. Désactiver temporairement le Service Worker
Le Service Worker pourrait causer des problèmes de cache. Pour tester :

1. Ouvrez la console du navigateur (F12)
2. Allez dans **Application** → **Service Workers**
3. Cliquez sur **Unregister** pour désactiver le SW
4. Rechargez la page

### 3. Vérifier les logs Vercel
Dans le dashboard Vercel :
- Allez dans **Deployments**
- Cliquez sur le dernier déploiement
- Vérifiez les **Function Logs** pour voir s'il y a des erreurs

### 4. Tester en local
```bash
npm run build
npm start
```
Si ça fonctionne en local mais pas sur Vercel, c'est un problème de configuration Vercel.

### 5. Vérifier les variables d'environnement
Dans Vercel :
- **Settings** → **Environment Variables**
- Vérifiez que toutes les variables sont correctement configurées
- Notamment `NEXT_PUBLIC_SUPABASE_URL` et `NEXT_PUBLIC_SUPABASE_ANON_KEY`

### 6. Désactiver le cache Next.js
Ajoutez dans `vercel.json` :
```json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "no-store, no-cache, must-revalidate"
        }
      ]
    }
  ]
}
```

### 7. Vérifier les logs du navigateur
Ouvrez la console (F12) et cherchez :
- `[DEBUG] 🔴 RELOAD` - Indique un reload automatique
- `[AuthProvider] Event:` - Indique les événements d'authentification
- Erreurs en rouge

## Diagnostic automatique
Le code inclut maintenant des logs de debug qui s'affichent dans la console du navigateur.

## Si le problème persiste
1. Partagez les logs de la console
2. Partagez les logs Vercel (Function Logs)
3. Indiquez si le problème existe aussi en local (`npm run dev`)

