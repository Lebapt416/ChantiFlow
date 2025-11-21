# Vérifier pourquoi les emails ne sont pas envoyés

## 1. Vérifier la configuration Resend sur Vercel

1. **Allez sur Vercel** : https://vercel.com
2. **Votre projet** → **Settings** → **Environment Variables**
3. **Vérifiez** que ces variables existent :
   - ✅ `RESEND_API_KEY` : doit commencer par `re_`
   - ✅ `RESEND_FROM_EMAIL` : ex: `ChantiFlow <noreply@chantiflow.com>` ou `ChantiFlow <onboarding@resend.dev>`

## 2. Vérifier les logs Vercel

1. **Vercel** → **Votre projet** → **Deployments** → **Dernier déploiement** → **Logs**
2. **Cherchez** ces messages :
   - `📧 Tentative d'envoi email de bienvenue à:`
   - `✅ Email de bienvenue envoyé avec succès`
   - `⚠️ Email non envoyé:`
   - `❌ Erreur Resend`
   - `RESEND_API_KEY non configuré`

## 3. Scénarios possibles

### Scénario A : RESEND_API_KEY non configuré
**Logs à chercher** : `RESEND_API_KEY non configuré` ou `Resend non initialisé`

**Solution** :
1. Créez un compte sur https://resend.com
2. Créez une API Key
3. Ajoutez-la sur Vercel comme variable d'environnement
4. Redéployez

### Scénario B : RESEND_API_KEY invalide
**Logs à chercher** : `❌ Erreur Resend` avec un message d'erreur

**Solution** :
1. Vérifiez que la clé API est correcte
2. Vérifiez qu'elle n'a pas expiré
3. Créez une nouvelle clé API si nécessaire

### Scénario C : Email dans les spams
**Logs à chercher** : `✅ Email de bienvenue envoyé avec succès`

**Solution** :
1. Vérifiez votre dossier spam
2. Vérifiez que l'email du worker est correct
3. Si vous utilisez `onboarding@resend.dev`, vérifiez votre domaine sur Resend

### Scénario D : RESEND_FROM_EMAIL non configuré
**Logs à chercher** : Email envoyé mais peut-être rejeté

**Solution** :
1. Configurez `RESEND_FROM_EMAIL` sur Vercel
2. Pour tester : `ChantiFlow <onboarding@resend.dev>`
3. Pour production : Vérifiez votre domaine sur Resend et utilisez votre domaine

## 4. Test rapide

Pour tester si Resend fonctionne, ajoutez un worker avec votre propre email et vérifiez :
1. Les logs Vercel
2. Votre boîte mail (et spam)
3. Le dashboard Resend (section Logs)

## 5. Vérifier le dashboard Resend

1. Allez sur https://resend.com
2. **Dashboard** → **Logs**
3. Vérifiez si les emails apparaissent :
   - ✅ **Sent** : Email envoyé avec succès
   - ⚠️ **Failed** : Email échoué (voir la raison)
   - 📧 **Pending** : Email en attente

## 6. Solutions selon les erreurs

### "Invalid API key"
→ Vérifiez que `RESEND_API_KEY` est correcte et commence par `re_`

### "Domain not verified"
→ Utilisez `onboarding@resend.dev` pour tester, ou vérifiez votre domaine sur Resend

### "Rate limit exceeded"
→ Vous avez atteint la limite (100 emails/jour en gratuit). Attendez ou upgradez.

### Aucun log dans Vercel
→ Vérifiez que le code a bien été déployé et que les variables d'environnement sont bien configurées.

## 7. Test manuel

Pour tester manuellement l'envoi d'email, vous pouvez créer une route API de test (optionnel) :

```typescript
// src/app/api/test-email/route.ts
import { sendWorkerWelcomeEmail } from '@/lib/email';

export async function GET() {
  const result = await sendWorkerWelcomeEmail({
    workerEmail: 'votre-email@example.com',
    workerName: 'Test',
  });
  return Response.json(result);
}
```

Puis visitez `/api/test-email` pour tester.

