# Guide de débogage - Envoi d'email de confirmation

## Vérifications à faire

### 1. Vérifier que RESEND_API_KEY est configuré

**En local :**
- Vérifiez que le fichier `.env.local` contient `RESEND_API_KEY=re_...`
- Redémarrez le serveur de développement après avoir ajouté la variable

**Sur Vercel :**
- Allez dans les paramètres du projet Vercel
- Section "Environment Variables"
- Vérifiez que `RESEND_API_KEY` est bien configuré
- Redéployez l'application après avoir ajouté/modifié la variable

### 2. Vérifier les logs dans la console du navigateur

1. Ouvrez la console du navigateur (F12)
2. Essayez de vous ajouter via le QR code
3. Regardez les logs qui commencent par 📧

Vous devriez voir :
- `📧 Tentative d'envoi email de confirmation à: [email]`
- `📧 Données envoyées: { workerEmail, workerName, userId }`
- `📧 Réponse API status: 200 OK` (ou une erreur)
- `✅ Email de confirmation envoyé avec succès` (si ça fonctionne)

### 3. Vérifier les logs serveur

**En local :**
- Regardez les logs dans votre terminal où tourne `npm run dev`

**Sur Vercel :**
- Allez dans "Functions" > "Logs" dans le dashboard Vercel
- Cherchez les logs qui commencent par 📧

### 4. Vérifier que l'email est bien rempli

- Assurez-vous que le champ email dans le formulaire est bien rempli
- L'email n'est pas obligatoire, mais s'il n'est pas rempli, aucun email ne sera envoyé

### 5. Tester l'API directement

Vous pouvez tester l'API directement avec curl :

```bash
curl -X POST http://localhost:3000/api/team/join-confirmation \
  -H "Content-Type: application/json" \
  -d '{
    "workerEmail": "test@example.com",
    "workerName": "Test User",
    "userId": "VOTRE_USER_ID"
  }'
```

Remplacez `VOTRE_USER_ID` par l'ID d'un utilisateur valide de votre base de données.

### 6. Vérifier la configuration Resend

- Vérifiez que votre compte Resend est actif
- Vérifiez que le domaine d'envoi est bien configuré dans Resend
- Vérifiez que `RESEND_FROM_EMAIL` est configuré (optionnel, utilise `onboarding@resend.dev` par défaut)

## Messages d'erreur courants

### "RESEND_API_KEY non configuré"
→ Ajoutez `RESEND_API_KEY` dans vos variables d'environnement

### "Service email non initialisé"
→ Vérifiez que `RESEND_API_KEY` est valide et que Resend est correctement initialisé

### "Email non fourni"
→ L'email n'a pas été rempli dans le formulaire

### Erreur HTTP 500
→ Vérifiez les logs serveur pour plus de détails

## Prochaines étapes

Si après ces vérifications le problème persiste :
1. Partagez les logs de la console du navigateur
2. Partagez les logs serveur (Vercel ou terminal local)
3. Vérifiez que `RESEND_API_KEY` est bien configuré et valide

