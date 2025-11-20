# Guide de déploiement - ChantiFlow

Ce guide vous explique comment déployer ChantiFlow sur Vercel et connecter votre nom de domaine `chantiflow.com`.

## Prérequis

- ✅ Un compte Vercel (gratuit) : [vercel.com/signup](https://vercel.com/signup)
- ✅ Un compte Supabase (gratuit) : [supabase.com](https://supabase.com)
- ✅ Votre nom de domaine `chantiflow.com` configuré chez Hostinger
- ✅ Git installé sur votre machine

## Étape 1 : Préparer le code sur GitHub

1. **Créer un dépôt GitHub** (si ce n'est pas déjà fait) :
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/VOTRE_USERNAME/chantiflow.git
   git push -u origin main
   ```

2. **Créer un fichier `.gitignore`** (si pas déjà présent) :
   ```
   .env.local
   .next
   node_modules
   .DS_Store
   ```

## Étape 2 : Configurer Supabase pour la production

1. **Aller sur votre projet Supabase** : [app.supabase.com](https://app.supabase.com)

2. **Configurer les URLs autorisées** :
   - Allez dans **Settings** → **Authentication** → **URL Configuration**
   - Ajoutez dans **Site URL** : `https://chantiflow.com`
   - Ajoutez dans **Redirect URLs** :
     - `https://chantiflow.com`
     - `https://chantiflow.com/login`
     - `https://www.chantiflow.com` (si vous utilisez www)
     - `https://www.chantiflow.com/login`

3. **Récupérer vos clés Supabase** :
   - **Settings** → **API**
   - Copiez :
     - `Project URL` → `NEXT_PUBLIC_SUPABASE_URL`
     - `anon public` key → `NEXT_PUBLIC_SUPABASE_ANON_KEY`
     - `service_role` key → `SUPABASE_SERVICE_ROLE_KEY` (⚠️ gardez-la secrète)

## Étape 3 : Déployer sur Vercel

1. **Connecter votre dépôt GitHub à Vercel** :
   - Allez sur [vercel.com/new](https://vercel.com/new)
   - Cliquez sur **Import Git Repository**
   - Sélectionnez votre dépôt `chantiflow`
   - Cliquez sur **Import**

2. **Configurer les variables d'environnement** :
   Dans la section **Environment Variables**, ajoutez :
   
   ```
   NEXT_PUBLIC_APP_BASE_URL=https://chantiflow.com
   NEXT_PUBLIC_SUPABASE_URL=votre_url_supabase
   NEXT_PUBLIC_SUPABASE_ANON_KEY=votre_anon_key
   SUPABASE_SERVICE_ROLE_KEY=votre_service_role_key
   ```

3. **Déployer** :
   - Cliquez sur **Deploy**
   - Attendez 2-3 minutes que le déploiement se termine
   - Votre site sera accessible sur `votre-projet.vercel.app`

## Étape 4 : Connecter votre nom de domaine

1. **Dans Vercel** :
   - Allez dans votre projet → **Settings** → **Domains**
   - Cliquez sur **Add Domain**
   - Entrez `chantiflow.com`
   - Cliquez sur **Add**

2. **Configurer les DNS chez Hostinger** :
   - Connectez-vous à votre compte Hostinger
   - Allez dans **Noms de domaine** → **chantiflow.com** → **Gérer les DNS**
   - Vercel vous donnera des enregistrements DNS à ajouter :
     - **Type A** : `@` → `76.76.21.21` (ou l'IP fournie par Vercel)
     - **Type CNAME** : `www` → `cname.vercel-dns.com` (ou la valeur fournie)

   ⚠️ **Important** : Supprimez les anciens enregistrements A/CNAME qui pointent ailleurs.

3. **Attendre la propagation DNS** :
   - Cela peut prendre de 5 minutes à 48 heures
   - Vérifiez avec : [whatsmydns.net](https://www.whatsmydns.net/#A/chantiflow.com)

4. **Vérifier le certificat SSL** :
   - Vercel génère automatiquement un certificat SSL (HTTPS)
   - Vérifiez que `https://chantiflow.com` fonctionne

## Étape 5 : Vérifier le déploiement

1. **Tester l'authentification** :
   - Visitez `https://chantiflow.com/login`
   - Connectez-vous avec votre compte Supabase

2. **Tester les QR codes** :
   - Créez un chantier
   - Vérifiez que le QR code pointe vers `https://chantiflow.com/qr/[siteId]`

3. **Vérifier les images** :
   - Testez l'upload de photos dans les rapports
   - Vérifiez que les images s'affichent correctement

## Dépannage

### Le domaine ne fonctionne pas
- Vérifiez les DNS avec [whatsmydns.net](https://www.whatsmydns.net)
- Assurez-vous que les enregistrements DNS sont corrects chez Hostinger
- Attendez jusqu'à 48h pour la propagation complète

### Erreur d'authentification
- Vérifiez que les URLs sont bien configurées dans Supabase
- Vérifiez que `NEXT_PUBLIC_APP_BASE_URL` est bien `https://chantiflow.com` dans Vercel

### Les images ne s'affichent pas
- Vérifiez que le bucket `reports` existe dans Supabase Storage
- Vérifiez les politiques RLS du bucket

### Erreur 500 en production
- Vérifiez les logs dans Vercel : **Deployments** → Cliquez sur un déploiement → **Functions** → **View Logs**
- Vérifiez que toutes les variables d'environnement sont bien configurées

## Mises à jour futures

À chaque `git push` sur la branche `main`, Vercel redéploie automatiquement votre site.

Pour forcer un redéploiement :
- Allez dans **Deployments** → Cliquez sur **...** → **Redeploy**

## Support

- Documentation Vercel : [vercel.com/docs](https://vercel.com/docs)
- Documentation Supabase : [supabase.com/docs](https://supabase.com/docs)
- Support Vercel : [vercel.com/support](https://vercel.com/support)

