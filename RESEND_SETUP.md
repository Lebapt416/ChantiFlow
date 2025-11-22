# Configuration Resend pour envoyer des emails à n'importe qui

## ⚠️ Important

**Pour envoyer des emails à n'importe qui avec Resend, vous DEVEZ vérifier un domaine.** C'est une exigence de sécurité de Resend et il n'y a pas de moyen de contourner cela.

## Solution : Vérifier un domaine dans Resend (OBLIGATOIRE)

### Étape 1 : Connectez-vous à Resend

1. Allez sur https://resend.com
2. Connectez-vous à votre compte (ou créez-en un si nécessaire)

### Étape 2 : Ajoutez votre domaine

1. Dans le menu de gauche, cliquez sur **"Domains"**
2. Cliquez sur **"Add Domain"** (ou "Ajouter un domaine")
3. Entrez votre domaine (ex: `chantiflow.com` ou `mail.chantiflow.com`)
   - ⚠️ **Important** : Vous devez posséder ce domaine et avoir accès à ses paramètres DNS

### Étape 3 : Configurez les enregistrements DNS

Resend vous donnera **3 enregistrements DNS** à ajouter dans votre gestionnaire de domaine :

#### A. Enregistrement TXT (Vérification)
- **Type** : TXT
- **Nom** : `@` ou votre sous-domaine
- **Valeur** : Copiez la valeur fournie par Resend
- **TTL** : 3600 (ou la valeur par défaut)

#### B. Enregistrement SPF
- **Type** : TXT
- **Nom** : `@` ou votre sous-domaine
- **Valeur** : `v=spf1 include:resend.com ~all`
- **TTL** : 3600

#### C. Enregistrements DKIM (2 enregistrements)
Resend vous donnera 2 clés DKIM à ajouter :
- **Type** : TXT
- **Nom** : `resend._domainkey` (ou similaire, selon ce que Resend indique)
- **Valeur** : Copiez la valeur fournie par Resend
- **TTL** : 3600

### Étape 4 : Attendez la vérification

1. Après avoir ajouté les enregistrements DNS, retournez sur Resend
2. Cliquez sur **"Verify"** (ou "Vérifier")
3. La vérification peut prendre **quelques minutes à 24 heures** selon votre fournisseur DNS
4. Vous recevrez un email de confirmation une fois le domaine vérifié

### Étape 5 : Configurez les variables d'environnement

Une fois le domaine vérifié, mettez à jour vos variables d'environnement :

**En local (`.env.local`)** :
```env
RESEND_API_KEY=re_xxxxxxxxxxxxx
RESEND_FROM_EMAIL=ChantiFlow <noreply@votredomaine.com>
```

**Sur Vercel** :
1. Allez dans **Settings** > **Environment Variables**
2. Ajoutez/modifiez :
   - `RESEND_API_KEY` : Votre clé API Resend
   - `RESEND_FROM_EMAIL` : `ChantiFlow <noreply@votredomaine.com>` (remplacez par votre domaine vérifié)

### Étape 6 : Redéployez

Après avoir mis à jour les variables d'environnement :
- **En local** : Redémarrez le serveur (`npm run dev`)
- **Sur Vercel** : Redéployez l'application

## 🎯 Résultat

Une fois le domaine vérifié, vous pourrez envoyer des emails à **n'importe quelle adresse email** !

## 📝 Notes importantes

- **Vous devez posséder le domaine** : Vous ne pouvez pas utiliser un domaine que vous ne possédez pas
- **Sous-domaine possible** : Vous pouvez utiliser un sous-domaine (ex: `mail.chantiflow.com`) au lieu du domaine principal
- **Gratuit** : La vérification de domaine est gratuite avec Resend
- **Pas de contournement** : Il n'existe pas de moyen de contourner cette exigence avec Resend

## 🔍 Vérification que ça fonctionne

Une fois le domaine vérifié et les variables configurées :

1. **Testez l'envoi d'email** avec une adresse email différente de la vôtre
2. **Vérifiez les logs** dans la console du navigateur (F12) et les logs serveur
3. **Vérifiez la boîte mail** du destinataire (y compris les spams)

## ❌ Messages d'erreur courants

### "not authorized" ou "not allowed"
→ **Le domaine n'est pas encore vérifié** ou `RESEND_FROM_EMAIL` n'utilise pas le domaine vérifié.

**Solution** : Vérifiez que :
- Le domaine est bien vérifié dans Resend (statut "Verified")
- `RESEND_FROM_EMAIL` utilise le domaine vérifié (ex: `noreply@votredomaine.com`)

### "domain not verified"
→ Le domaine utilisé dans `RESEND_FROM_EMAIL` n'est pas vérifié dans Resend.

**Solution** : Vérifiez le domaine dans Resend et attendez la vérification complète.

### "DNS records not found"
→ Les enregistrements DNS n'ont pas encore été propagés.

**Solution** : Attendez quelques heures et réessayez. La propagation DNS peut prendre jusqu'à 24-48 heures.

## 🆘 Besoin d'aide ?

- **Documentation Resend** : https://resend.com/docs
- **Support Resend** : support@resend.com
- **Vérification DNS** : Utilisez des outils comme https://mxtoolbox.com pour vérifier vos enregistrements DNS

## Documentation Resend

- [Documentation Resend](https://resend.com/docs)
- [Vérification de domaine](https://resend.com/docs/dashboard/domains/introduction)
- [Configuration DNS](https://resend.com/docs/dashboard/domains/verify-a-domain)

