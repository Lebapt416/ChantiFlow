# Configuration OpenAI pour ChantiFlow

## 🔍 Vérifier que votre API key est bien configurée

### 1. Vérifier dans Vercel

1. Allez sur [vercel.com](https://vercel.com)
2. Ouvrez votre projet `chatiflovaz`
3. Allez dans **Settings → Environment Variables**
4. Vérifiez que `OPENAI_API_KEY` existe et contient votre clé API

### 2. Vérifier le format de la clé

Votre clé OpenAI doit :
- Commencer par `sk-`
- Avoir environ 51 caractères
- Ne pas contenir d'espaces avant/après

Exemple : `sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### 3. Redéployer après ajout/modification

⚠️ **Important** : Après avoir ajouté ou modifié `OPENAI_API_KEY` dans Vercel, vous devez **redéployer** :

1. Allez dans **Deployments**
2. Cliquez sur **"..."** sur le dernier déploiement
3. Cliquez sur **"Redeploy"**

### 4. Vérifier les logs

Si ça ne fonctionne toujours pas :

1. Allez dans **Deployments**
2. Cliquez sur le dernier déploiement
3. Allez dans **"Functions"** ou **"Runtime Logs"**
4. Cherchez les logs qui commencent par `[AI Planning]`
5. Vous verrez :
   - `[AI Planning] Pas d'API key OpenAI` → La clé n'est pas accessible
   - `[AI Planning] Appel OpenAI avec X tâches` → L'appel est fait
   - `[AI Planning] Erreur OpenAI:` → Il y a une erreur (détails après)

## 🐛 Erreurs courantes

### "Pas d'API key OpenAI"
- La variable `OPENAI_API_KEY` n'existe pas dans Vercel
- Ou elle est vide
- **Solution** : Ajoutez-la et redéployez

### "OpenAI API error: 401"
- La clé API est invalide ou expirée
- **Solution** : Vérifiez votre clé sur [platform.openai.com](https://platform.openai.com/api-keys)

### "OpenAI API error: 429"
- Vous avez dépassé votre quota
- **Solution** : Vérifiez votre quota sur [platform.openai.com](https://platform.openai.com/usage)

### "Erreur lors du parsing de la réponse OpenAI"
- L'API a retourné un format JSON invalide
- **Solution** : Réessayez, c'est généralement temporaire

## 📝 Créer une clé API OpenAI

Si vous n'avez pas encore de clé :

1. Allez sur [platform.openai.com](https://platform.openai.com)
2. Créez un compte ou connectez-vous
3. Allez dans **API keys**
4. Cliquez sur **"Create new secret key"**
5. Copiez la clé (vous ne pourrez plus la voir après)
6. Ajoutez-la dans Vercel comme `OPENAI_API_KEY`

## 💡 Astuce

Pour tester rapidement si votre clé fonctionne, vous pouvez vérifier les logs Vercel après avoir généré un planning. Si vous voyez `[AI Planning] Planning généré avec succès`, c'est que tout fonctionne !

