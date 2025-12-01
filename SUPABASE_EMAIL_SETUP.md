# Configuration des emails de réinitialisation de mot de passe

Pour que les emails de réinitialisation de mot de passe fonctionnent, vous devez configurer Supabase correctement.

## 1. Autoriser l'URL de redirection

1. Allez dans votre **Supabase Dashboard**
2. Naviguez vers **Authentication** > **URL Configuration**
3. Dans la section **Redirect URLs**, ajoutez **les deux URLs** :
   
   **Production :**
   ```
   https://votre-domaine.com/api/auth/reset-password
   ```
   
   **Développement local :**
   ```
   http://localhost:3000/api/auth/reset-password
   ```
   
   **Important** : 
   - Les deux URLs doivent être ajoutées pour pouvoir tester en local ET en production
   - Le code détecte automatiquement l'URL actuelle (localhost ou production)
   - Si vous testez en local, assurez-vous que `NEXT_PUBLIC_APP_BASE_URL=http://localhost:3000` dans votre `.env.local`

## 2. Configurer le service d'email

### Option A : Utiliser le service email par défaut de Supabase (limité)

Par défaut, Supabase envoie les emails via son propre service, mais il y a des limites :
- Maximum 3 emails par heure par utilisateur
- Les emails peuvent être marqués comme spam

### Option B : Configurer un service SMTP personnalisé (recommandé)

1. Allez dans **Project Settings** > **Auth** > **SMTP Settings**
2. Activez **Enable Custom SMTP**
3. Configurez votre service SMTP (Gmail, SendGrid, Mailgun, etc.)

**Exemple avec Gmail :**
- **Host**: `smtp.gmail.com`
- **Port**: `587`
- **Username**: Votre adresse Gmail
- **Password**: Un mot de passe d'application Gmail (pas votre mot de passe normal)
- **Sender email**: Votre adresse Gmail
- **Sender name**: ChantiFlow

## 3. Personnaliser les templates d'email (optionnel)

1. Allez dans **Authentication** > **Email Templates**
2. Sélectionnez **Reset Password**
3. Personnalisez le sujet et le contenu de l'email
4. Assurez-vous que le lien de réinitialisation contient : `{{ .ConfirmationURL }}`

## 4. Vérifier les logs

Si les emails ne sont toujours pas envoyés :

1. Allez dans **Logs** > **Auth Logs** dans Supabase Dashboard
2. Vérifiez les erreurs liées à l'envoi d'emails
3. Vérifiez les logs de votre application (console.log dans `resetPasswordRequestAction`)

## 5. Variables d'environnement

Assurez-vous que `NEXT_PUBLIC_APP_BASE_URL` est correctement configuré :

```env
NEXT_PUBLIC_APP_BASE_URL=https://votre-domaine.com
```

Pour le développement local :
```env
NEXT_PUBLIC_APP_BASE_URL=http://localhost:3000
```

## Dépannage

### Les emails ne sont pas reçus

1. ✅ Vérifiez vos **spams/courriers indésirables**
2. ✅ Vérifiez que l'URL de redirection est autorisée dans Supabase
3. ✅ Vérifiez que le service SMTP est configuré (si vous utilisez un service personnalisé)
4. ✅ Vérifiez les logs Supabase pour les erreurs
5. ✅ Vérifiez que `NEXT_PUBLIC_APP_BASE_URL` est correctement défini

### Erreur "rate limit exceeded"

- Supabase limite le nombre d'emails par heure
- Attendez quelques minutes avant de réessayer
- Ou configurez un service SMTP personnalisé pour augmenter les limites

### Le lien de réinitialisation ne fonctionne pas

1. ✅ Vérifiez que l'URL de redirection est exactement la même dans le code et dans Supabase (`/api/auth/reset-password`)
2. ✅ Vérifiez que le lien n'a pas expiré (les liens expirent après 1 heure par défaut)
3. ✅ Vérifiez les logs de l'application pour voir les erreurs
4. ✅ Vérifiez la console du navigateur pour les erreurs JavaScript

### Test en localhost

Si vous testez en localhost mais que le lien pointe vers la production :

1. **Option 1 (Recommandé)** : Assurez-vous que `http://localhost:3000/api/auth/reset-password` est dans les URLs autorisées de Supabase, puis :
   - Modifiez temporairement `NEXT_PUBLIC_APP_BASE_URL=http://localhost:3000` dans `.env.local`
   - Redémarrez le serveur Next.js
   - Demandez un nouveau lien de réinitialisation

2. **Option 2** : Copiez le code depuis l'URL de production :
   - Cliquez sur le lien dans l'email (il ouvrira la production)
   - Copiez le paramètre `code` dans l'URL (ex: `?code=abc123&type=recovery`)
   - Allez sur `http://localhost:3000/api/auth/reset-password?code=abc123&type=recovery`
   - Vous serez redirigé vers la page de réinitialisation en local

