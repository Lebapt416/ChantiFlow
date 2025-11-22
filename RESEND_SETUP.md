# Configuration Resend pour l'envoi d'emails

## Problème actuel

Resend en mode développement/test ne peut envoyer des emails qu'à l'adresse email vérifiée de votre compte Resend. C'est une limitation de sécurité pour éviter l'abus.

## Solutions

### Option 1 : Vérifier un domaine dans Resend (Recommandé pour la production)

1. **Connectez-vous à votre compte Resend** : https://resend.com
2. **Allez dans "Domains"** dans le menu
3. **Ajoutez votre domaine** (ex: `chantiflow.com`)
4. **Configurez les enregistrements DNS** comme indiqué par Resend :
   - Un enregistrement TXT pour la vérification
   - Un enregistrement SPF
   - Un enregistrement DKIM
5. **Attendez la vérification** (peut prendre quelques minutes à quelques heures)
6. **Mettez à jour `RESEND_FROM_EMAIL`** dans vos variables d'environnement :
   ```
   RESEND_FROM_EMAIL=ChantiFlow <noreply@votredomaine.com>
   ```

### Option 2 : Utiliser l'email de test pour le développement

Pour tester en local, vous pouvez temporairement utiliser votre propre email vérifié :

1. **Récupérez votre email vérifié** depuis votre compte Resend
2. **Utilisez cet email** pour tester les fonctionnalités
3. **En production**, utilisez un domaine vérifié (Option 1)

### Option 3 : Passer à un plan payant Resend

Les plans payants Resend offrent plus de flexibilité, mais la vérification de domaine reste recommandée pour la production.

## Configuration des variables d'environnement

### En local (`.env.local`)
```env
RESEND_API_KEY=re_xxxxxxxxxxxxx
RESEND_FROM_EMAIL=ChantiFlow <noreply@votredomaine.com>
```

### Sur Vercel
1. Allez dans **Settings** > **Environment Variables**
2. Ajoutez :
   - `RESEND_API_KEY` : Votre clé API Resend
   - `RESEND_FROM_EMAIL` : L'email d'envoi (doit être depuis un domaine vérifié en production)

## Vérification

Pour vérifier que votre configuration fonctionne :

1. **Testez avec votre email vérifié** (dev/test)
2. **Vérifiez les logs** dans la console du navigateur et les logs serveur
3. **Vérifiez votre boîte mail** (y compris les spams)

## Messages d'erreur courants

### "not authorized" ou "not allowed"
→ L'email de destination n'est pas autorisé. En mode dev, seul l'email vérifié du compte Resend peut recevoir des emails.

### "domain not verified"
→ Le domaine utilisé dans `RESEND_FROM_EMAIL` n'est pas vérifié dans Resend.

### Solution temporaire pour le développement

Si vous voulez tester rapidement sans configurer un domaine, vous pouvez :
1. Utiliser votre email vérifié comme destinataire de test
2. Ou modifier temporairement le code pour forcer l'envoi à votre email (non recommandé pour la production)

## Documentation Resend

- [Documentation Resend](https://resend.com/docs)
- [Vérification de domaine](https://resend.com/docs/dashboard/domains/introduction)
- [Configuration DNS](https://resend.com/docs/dashboard/domains/verify-a-domain)

