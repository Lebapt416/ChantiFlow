# Test de l'API Supabase

## Vérifications à faire

### 1. Variables d'environnement
Vérifiez que ces variables sont bien configurées dans Vercel :
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY` (pour les Server Actions)

### 2. Test de connexion
Ouvrez la console du navigateur (F12) et vérifiez :
- Pas d'erreurs `Supabase est mal configuré`
- Les logs `✅ Session restaurée` ou `❌ Aucune session trouvée` apparaissent
- Pas d'erreurs réseau dans l'onglet Network

### 3. Test d'authentification
1. Allez sur `/login`
2. Connectez-vous
3. Vérifiez que vous êtes redirigé vers `/home`
4. Vérifiez que vos données s'affichent

### 4. Test des requêtes
Ouvrez DevTools → Network et vérifiez :
- Les requêtes vers `*.supabase.co` fonctionnent
- Pas d'erreurs 401 (Unauthorized)
- Pas d'erreurs 500 (Server Error)

### 5. Logs Vercel
Vérifiez les logs de déploiement Vercel :
- Pas d'erreurs de build liées à Supabase
- Les variables d'environnement sont bien chargées

## Problèmes courants

### "Supabase est mal configuré"
- Vérifiez que les variables sont dans Vercel (Settings → Environment Variables)
- Redéployez après avoir ajouté les variables

### Erreurs 401
- La clé `NEXT_PUBLIC_SUPABASE_ANON_KEY` est invalide
- Récupérez la bonne clé depuis le dashboard Supabase

### Erreurs de réseau
- Vérifiez que votre projet Supabase est actif
- Vérifiez les restrictions IP dans Supabase (Settings → API)

## Code vérifié

✅ `createSupabaseBrowserClient()` - Fonctionne correctement
✅ `createSupabaseServerClient()` - Fonctionne correctement  
✅ `AuthProvider` - Utilise Supabase correctement (non-bloquant)
✅ Toutes les pages utilisent les clients Supabase correctement

