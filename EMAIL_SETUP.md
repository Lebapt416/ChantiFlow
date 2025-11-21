# Configuration de l'envoi d'emails avec Resend

## 1. Créer un compte Resend

1. Allez sur https://resend.com
2. Créez un compte gratuit (100 emails/jour)
3. Vérifiez votre email

## 2. Créer une API Key

1. Dans le dashboard Resend, allez dans **API Keys**
2. Cliquez sur **Create API Key**
3. Donnez un nom (ex: "ChantiFlow Production")
4. Copiez la clé API (elle ne sera affichée qu'une seule fois)

## 3. Ajouter les variables d'environnement

### Localement (`.env.local`)

```bash
RESEND_API_KEY=re_xxxxxxxxxxxxx
RESEND_FROM_EMAIL=ChantiFlow <noreply@votre-domaine.com>
NEXT_PUBLIC_APP_BASE_URL=http://localhost:3000
```

### Sur Vercel

1. Allez dans votre projet Vercel
2. **Settings** → **Environment Variables**
3. Ajoutez :
   - `RESEND_API_KEY` : votre clé API Resend
   - `RESEND_FROM_EMAIL` : `ChantiFlow <noreply@votre-domaine.com>` (remplacez par votre domaine vérifié)
   - `NEXT_PUBLIC_APP_BASE_URL` : `https://chantiflow.com` (ou votre domaine)

## 4. Vérifier un domaine (optionnel mais recommandé)

Pour envoyer depuis votre propre domaine :

1. Dans Resend, allez dans **Domains**
2. Cliquez sur **Add Domain**
3. Ajoutez votre domaine (ex: `chantiflow.com`)
4. Ajoutez les enregistrements DNS fournis dans votre registrar
5. Attendez la vérification (peut prendre quelques minutes)

Une fois vérifié, utilisez : `ChantiFlow <noreply@chantiflow.com>`

## 5. Tester

1. Ajoutez un worker avec un email valide depuis `/team` ou `/site/[id]`
2. Vérifiez que l'email de bienvenue est reçu
3. Envoyez un rapport depuis le QR code
4. Vérifiez que le manager reçoit la notification

## Emails envoyés

### Email de bienvenue (worker)
- **Quand** : Lorsqu'un worker est ajouté à l'équipe ou à un chantier
- **Contenu** : Message de bienvenue avec lien vers le formulaire de rapport
- **Destinataire** : L'email du worker

### Email de notification (manager)
- **Quand** : Lorsqu'un worker envoie un rapport
- **Contenu** : Notification avec lien vers les rapports
- **Destinataire** : L'email du manager du chantier

## Dépannage

### Les emails ne sont pas envoyés

1. Vérifiez que `RESEND_API_KEY` est bien configuré
2. Vérifiez les logs dans la console Vercel
3. Vérifiez que l'email du destinataire est valide
4. En mode développement, les erreurs sont loggées dans la console

### Erreur "Domain not verified"

- Utilisez l'email par défaut de Resend : `onboarding@resend.dev`
- Ou vérifiez votre domaine dans Resend

