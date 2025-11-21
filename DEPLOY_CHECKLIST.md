# Checklist de déploiement - ChantiFlow

## ✅ Code prêt

- [x] Tous les changements sont commités
- [x] Code poussé sur GitHub
- [x] Build fonctionne localement

## 🔧 Variables d'environnement à configurer sur Vercel

### Variables Supabase (déjà configurées)
- ✅ `NEXT_PUBLIC_SUPABASE_URL`
- ✅ `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- ✅ `SUPABASE_SERVICE_ROLE_KEY`
- ✅ `NEXT_PUBLIC_APP_BASE_URL` (https://chantiflow.com)

### Variables Resend (NOUVELLES - À AJOUTER)

1. **Allez sur Vercel** → Votre projet → **Settings** → **Environment Variables**

2. **Ajoutez ces variables** :

   ```
   RESEND_API_KEY=re_xxxxxxxxxxxxx
   ```
   - Obtenez votre clé sur https://resend.com/api-keys
   - Créez un compte si nécessaire (gratuit, 100 emails/jour)

   ```
   RESEND_FROM_EMAIL=ChantiFlow <noreply@chantiflow.com>
   ```
   - Pour tester : `ChantiFlow <onboarding@resend.dev>`
   - Pour production : vérifiez votre domaine sur Resend et utilisez votre domaine

## 📧 Configuration Resend

### 1. Créer un compte Resend
- Allez sur https://resend.com
- Créez un compte (gratuit)
- Vérifiez votre email

### 2. Créer une API Key
- Dashboard → **API Keys** → **Create API Key**
- Donnez un nom : "ChantiFlow Production"
- Copiez la clé (commence par `re_`)

### 3. (Optionnel) Vérifier votre domaine
- Dashboard → **Domains** → **Add Domain**
- Ajoutez `chantiflow.com`
- Ajoutez les enregistrements DNS fournis
- Une fois vérifié, utilisez : `ChantiFlow <noreply@chantiflow.com>`

## 🗄️ Migration base de données

### Exécuter la migration workers

1. **Allez sur Supabase** : https://app.supabase.com
2. **SQL Editor** → **New Query**
3. **Copiez-collez** le contenu de `migration-workers-to-account.sql`
4. **Run** pour exécuter

Cette migration permet :
- Workers au niveau du compte (réutilisables)
- Workers peuvent être assignés à plusieurs chantiers

## 🚀 Déploiement Vercel

### Si le déploiement automatique ne se fait pas :

1. **Allez sur Vercel** : https://vercel.com
2. **Votre projet** → **Deployments**
3. Cliquez sur **Redeploy** sur le dernier déploiement
4. Ou faites un commit vide pour déclencher un nouveau déploiement :
   ```bash
   git commit --allow-empty -m "Trigger deployment"
   git push
   ```

## ✅ Vérifications post-déploiement

### 1. Vérifier que le site fonctionne
- [ ] https://chantiflow.com charge correctement
- [ ] Connexion fonctionne
- [ ] Création de chantier fonctionne

### 2. Tester les emails
- [ ] Ajouter un worker avec un email valide depuis `/team`
- [ ] Vérifier que l'email de bienvenue est reçu
- [ ] Envoyer un rapport depuis le QR code
- [ ] Vérifier que le manager reçoit la notification

### 3. Vérifier les workers
- [ ] Ajouter un worker depuis `/team` (au niveau du compte)
- [ ] Ajouter ce worker à un chantier depuis `/site/[id]`
- [ ] Vérifier que l'email est envoyé

## 📝 Logs et dépannage

### Vérifier les logs Vercel
- Vercel → Votre projet → **Deployments** → Cliquez sur un déploiement → **Logs**

### Vérifier les logs Resend
- Resend Dashboard → **Logs** → Voir les emails envoyés/échoués

### Erreurs courantes

**"RESEND_API_KEY non configuré"**
- Vérifiez que la variable est bien ajoutée sur Vercel
- Redéployez après avoir ajouté la variable

**"Domain not verified"**
- Utilisez `onboarding@resend.dev` pour tester
- Ou vérifiez votre domaine sur Resend

**Emails non reçus**
- Vérifiez les logs Resend
- Vérifiez que l'email du destinataire est valide
- Vérifiez les spams

## 🎉 C'est prêt !

Une fois tout configuré :
- Les workers recevront un email de bienvenue
- Les managers recevront des notifications lors des rapports
- Tout fonctionne automatiquement !

