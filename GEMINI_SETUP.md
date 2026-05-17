# Configuration Google Gemini pour ChantiFlow

Ce guide vous explique comment configurer l'API Google Gemini pour activer les fonctionnalités IA dans ChantiFlow.

## 📋 Prérequis

1. Un compte Google Cloud Platform (GCP)
2. Un projet GCP avec l'API Generative AI activée
3. Une clé API Gemini

## 🔑 Obtenir une clé API Gemini

1. Allez sur [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Connectez-vous avec votre compte Google
3. Cliquez sur "Create API Key"
4. Sélectionnez votre projet GCP (ou créez-en un nouveau)
5. Copiez la clé API générée

## ⚙️ Configuration

### 1. Variables d'environnement locales

Ajoutez la clé API dans votre fichier `.env.local` :

```env
GOOGLE_GEMINI_API_KEY=<VOTRE_CLE_API_ICI>
```

### 2. Configuration Vercel

1. Allez sur votre projet Vercel
2. Ouvrez **Settings** → **Environment Variables**
3. Ajoutez la variable :
   - **Name**: `GOOGLE_GEMINI_API_KEY`
   - **Value**: Votre clé API Gemini
   - **Environments**: Production, Preview, Development
4. Cliquez sur **Save**

⚠️ **Important** : Après avoir ajouté ou modifié `GOOGLE_GEMINI_API_KEY` dans Vercel, vous devez **redéployer** :
- Allez dans **Deployments**
- Cliquez sur les trois points (⋯) du dernier déploiement
- Sélectionnez **Redeploy**

## ✅ Vérification

Une fois configuré, vous devriez voir dans les logs :
- `[AI Planning] Appel Google Gemini avec X tâches` → L'appel est fait
- `[AI Planning] Réponse Gemini reçue` → Succès
- `[AI Planning] Erreur Gemini:` → Il y a une erreur (détails après)

## 🐛 Dépannage

### "Pas d'API key Google Gemini"
- La variable `GOOGLE_GEMINI_API_KEY` n'existe pas dans Vercel
- Vérifiez que vous avez bien redéployé après avoir ajouté la variable

### "Clé API Google Gemini invalide"
- La clé API est incorrecte ou expirée
- **Solution** : Vérifiez votre clé sur [Google AI Studio](https://aistudio.google.com/app/apikey)

### "Quota Google Gemini dépassé"
- Vous avez atteint la limite de requêtes gratuites
- **Solution** : Attendez quelques minutes ou vérifiez votre quota sur [Google Cloud Console](https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas)

### "Erreur lors du parsing de la réponse Gemini"
- La réponse n'est pas au format JSON attendu
- **Solution** : Vérifiez les logs pour voir la réponse complète

## 📝 Modèle utilisé

ChantiFlow utilise actuellement **Gemini 1.5 Pro** pour :
- La génération de plannings intelligents
- La génération de descriptions de tâches
- L'analyse et l'optimisation des chantiers

## 🔒 Sécurité

- ⚠️ **Ne commitez jamais** votre clé API dans le dépôt Git
- ✅ Utilisez toujours les variables d'environnement
- ✅ La clé API est stockée de manière sécurisée dans Vercel

## 📚 Ressources

- [Documentation Google Gemini](https://ai.google.dev/docs)
- [Google AI Studio](https://aistudio.google.com)
- [Tarification Gemini](https://ai.google.dev/pricing)

