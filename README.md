# ChantiFlow

## 🚀 Démarrage rapide de l'API IA

Pour utiliser les fonctionnalités IA (prédiction de durée, analyse de risque, recommandation d'équipe), vous devez démarrer le serveur FastAPI :

### Installation et démarrage

**Option 1 : Script automatique (macOS/Linux)**
```bash
cd ml
./start.sh
```

**Option 2 : Script automatique (Windows)**
```bash
cd ml
start.bat
```

**Option 3 : Manuel**
```bash
cd ml
pip3 install -r requirements.txt
python3 train_model.py  # Si predictor.pt n'existe pas
python3 api.py
```

L'API sera accessible sur `http://localhost:8000`

Pour plus de détails, consultez [ml/START_API.md](ml/START_API.md)

---

# ChantiFlow

Application web de gestion de chantiers avec planification IA et accès QR codes pour les employés.

## 🚀 Technologies

- **Next.js 16** (App Router)
- **Supabase** (Auth, Postgres, Storage)
- **Tailwind CSS**
- **TypeScript**
- **React 19**

## 📋 Fonctionnalités

- ✅ Création et gestion de chantiers
- ✅ Ajout de tâches et d'employés
- ✅ Génération de planning IA
- ✅ QR codes pour accès employés
- ✅ Upload de photos et rapports
- ✅ Tableau de bord avec statistiques
- ✅ Mode sombre/clair

## 🛠️ Installation locale

1. **Cloner le projet** :
   ```bash
   git clone https://github.com/VOTRE_USERNAME/chantiflow.git
   cd chantiflow
   ```

2. **Installer les dépendances** :
   ```bash
   npm install
   ```

3. **Configurer les variables d'environnement** :
   ```bash
   cp env.example .env.local
   ```
   
   Remplir `.env.local` avec vos clés Supabase :
   ```
   NEXT_PUBLIC_APP_BASE_URL=http://localhost:3000
   NEXT_PUBLIC_SUPABASE_URL=votre_url_supabase
   NEXT_PUBLIC_SUPABASE_ANON_KEY=votre_anon_key
   SUPABASE_SERVICE_ROLE_KEY=votre_service_role_key
   ```

4. **Créer les tables Supabase** :
   - Connectez-vous à [app.supabase.com](https://app.supabase.com)
   - Allez dans **SQL Editor**
   - Exécutez les scripts SQL pour créer les tables `sites`, `tasks`, `workers`, `reports`
   - Créez un bucket Storage nommé `reports` avec accès public

5. **Lancer le serveur de développement** :
   ```bash
   npm run dev
   ```

6. **Ouvrir dans le navigateur** :
   [http://localhost:3000](http://localhost:3000)

## 📦 Déploiement

Consultez le guide complet : **[DEPLOY.md](./DEPLOY.md)**

### Déploiement rapide sur Vercel

1. Créer un compte sur [vercel.com](https://vercel.com)
2. Importer le dépôt GitHub
3. Configurer les variables d'environnement
4. Connecter le domaine `chantiflow.com`

## 📚 Structure du projet

```
src/
├── app/              # Pages Next.js (App Router)
│   ├── dashboard/    # Tableau de bord
│   ├── site/[id]/    # Détails d'un chantier
│   ├── qr/[siteId]/  # Accès employé via QR
│   └── report/       # Rapports et photos
├── components/       # Composants React réutilisables
└── lib/             # Utilitaires (Supabase, etc.)
```

## 🔐 Sécurité

- Authentification via Supabase Auth
- Row Level Security (RLS) activé sur toutes les tables
- Variables d'environnement pour les clés sensibles
- HTTPS obligatoire en production

## 📝 Licence

Propriétaire - Tous droits réservés
