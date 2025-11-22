# Configuration Resend - Utiliser votre domaine vérifié

## ⚠️ Problème actuel

Votre domaine `chantiflow.com` est vérifié dans Resend, mais l'application utilise encore `onboarding@resend.dev` comme adresse d'envoi. Il faut configurer `RESEND_FROM_EMAIL` avec votre domaine vérifié.

## ✅ Solution : Configurer RESEND_FROM_EMAIL

### En local (`.env.local`)

1. Ouvrez le fichier `.env.local` à la racine du projet
2. Ajoutez ou modifiez la ligne suivante :

```env
RESEND_FROM_EMAIL=ChantiFlow <noreply@chantiflow.com>
```

**Format important** : `Nom <email@domaine-verifie.com>`

Vous pouvez utiliser :
- `noreply@chantiflow.com`
- `contact@chantiflow.com`
- `info@chantiflow.com`
- Ou n'importe quelle adresse avec le domaine `chantiflow.com`

### Sur Vercel

1. Allez sur https://vercel.com
2. Sélectionnez votre projet ChantiFlow
3. Allez dans **Settings** > **Environment Variables**
4. Cherchez `RESEND_FROM_EMAIL` :
   - Si elle existe, **modifiez-la**
   - Si elle n'existe pas, **ajoutez-la**
5. Valeur à mettre :
   ```
   ChantiFlow <noreply@chantiflow.com>
   ```
6. Sélectionnez **Production**, **Preview**, et **Development** (ou au moins Production)
7. Cliquez sur **Save**
8. **Redéployez votre application** :
   - Allez dans **Deployments**
   - Cliquez sur les 3 points (...) du dernier déploiement
   - Cliquez sur **Redeploy**

### Vérification

Après avoir configuré `RESEND_FROM_EMAIL` :

1. **En local** : Redémarrez le serveur (`npm run dev`)
2. **Sur Vercel** : Attendez la fin du redéploiement
3. **Testez l'envoi d'email** : Essayez de vous ajouter via le QR code
4. **Vérifiez les logs** : Dans la console du navigateur, vous devriez voir :
   ```
   📧 Resend: From email: ChantiFlow <noreply@chantiflow.com>
   ```

## 📝 Exemple de configuration complète

### `.env.local` (local)
```env
NEXT_PUBLIC_APP_BASE_URL=http://localhost:3000
NEXT_PUBLIC_SUPABASE_URL=votre_url_supabase
NEXT_PUBLIC_SUPABASE_ANON_KEY=votre_cle_anon
SUPABASE_SERVICE_ROLE_KEY=votre_service_role_key
OPENAI_API_KEY=votre_cle_openai
RESEND_API_KEY=re_xxxxxxxxxxxxx
RESEND_FROM_EMAIL=ChantiFlow <noreply@chantiflow.com>
```

### Variables d'environnement Vercel
- `RESEND_API_KEY` = `re_xxxxxxxxxxxxx`
- `RESEND_FROM_EMAIL` = `ChantiFlow <noreply@chantiflow.com>`

## ❌ Erreurs courantes

### "403 Forbidden" même après vérification du domaine
→ `RESEND_FROM_EMAIL` n'est pas configuré ou utilise encore `@resend.dev`

**Solution** : Vérifiez que `RESEND_FROM_EMAIL` utilise bien `@chantiflow.com`

### L'email utilise toujours `onboarding@resend.dev`
→ La variable d'environnement n'est pas chargée

**Solution** :
- Vérifiez que le fichier `.env.local` existe et contient `RESEND_FROM_EMAIL`
- Redémarrez le serveur de développement
- Sur Vercel, vérifiez que la variable est bien configurée et redéployez

## 🎯 Résultat attendu

Une fois configuré correctement, vous pourrez envoyer des emails à n'importe quelle adresse email, et l'adresse d'envoi sera `noreply@chantiflow.com` (ou celle que vous avez choisie).

