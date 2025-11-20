# Configuration Vercel - Variables d'environnement

## ⚠️ Erreur "Application error" ?

Cette erreur signifie que les **variables d'environnement Supabase ne sont pas configurées** sur Vercel.

## Solution rapide

### 1. Récupérer vos clés Supabase

1. Allez sur [app.supabase.com](https://app.supabase.com)
2. Sélectionnez votre projet
3. Allez dans **Settings** → **API**
4. Copiez ces 3 valeurs :
   - **Project URL** (ex: `https://xxxxx.supabase.co`)
   - **anon public** key (commence par `eyJ...`)
   - **service_role** key (commence par `eyJ...`) ⚠️ **GARDEZ-LA SECRÈTE**

### 2. Configurer sur Vercel

1. Allez sur [vercel.com](https://vercel.com)
2. Ouvrez votre projet **chatiflovaz**
3. Allez dans **Settings** → **Environment Variables**
4. Ajoutez ces 4 variables :

| Nom | Valeur | Environnement |
|-----|--------|---------------|
| `NEXT_PUBLIC_APP_BASE_URL` | `https://chatiflovaz.vercel.app` (ou votre domaine) | Production, Preview, Development |
| `NEXT_PUBLIC_SUPABASE_URL` | Votre Project URL de Supabase | Production, Preview, Development |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Votre anon public key | Production, Preview, Development |
| `SUPABASE_SERVICE_ROLE_KEY` | Votre service_role key | Production, Preview, Development |

### 3. Redéployer

1. Après avoir ajouté les variables, allez dans **Deployments**
2. Cliquez sur **...** sur le dernier déploiement
3. Cliquez sur **Redeploy**
4. Ou faites un nouveau commit et push sur GitHub

### 4. Vérifier

Une fois redéployé, votre site devrait fonctionner ! ✅

## Configuration Supabase pour la production

N'oubliez pas de configurer les URLs autorisées dans Supabase :

1. Allez dans **Settings** → **Authentication** → **URL Configuration**
2. Ajoutez dans **Site URL** : `https://chatiflovaz.vercel.app`
3. Ajoutez dans **Redirect URLs** :
   - `https://chatiflovaz.vercel.app`
   - `https://chatiflovaz.vercel.app/login`
   - `https://chatiflow.com` (si vous avez configuré le domaine)
   - `https://chatiflow.com/login`

## Dépannage

### L'erreur persiste après configuration

1. Vérifiez que toutes les variables sont bien ajoutées
2. Vérifiez que les valeurs sont correctes (pas d'espaces avant/après)
3. Vérifiez les logs Vercel : **Deployments** → Cliquez sur un déploiement → **Runtime Logs**

### Erreur "Invalid API key"

- Vérifiez que vous avez copié la bonne clé (anon vs service_role)
- Vérifiez qu'il n'y a pas d'espaces dans les valeurs

