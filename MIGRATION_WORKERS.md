# Migration : Workers au niveau du compte

Cette migration permet de stocker les workers au niveau du compte utilisateur, ce qui permet de les réutiliser sur plusieurs chantiers.

## Instructions

1. **Ouvrez l'éditeur SQL de Supabase** :
   - Allez sur https://app.supabase.com
   - Sélectionnez votre projet
   - Allez dans "SQL Editor"

2. **Exécutez la migration** :
   - Copiez le contenu du fichier `migration-workers-to-account.sql`
   - Collez-le dans l'éditeur SQL
   - Cliquez sur "Run" pour exécuter

3. **Vérifiez que la migration a réussi** :
   - Vérifiez qu'il n'y a pas d'erreurs dans la console
   - Les workers existants restent inchangés (ils gardent leur `site_id`)

## Changements apportés

- **Table `workers`** :
  - `site_id` devient nullable (peut être NULL pour les workers au niveau du compte)
  - `email` devient nullable
  - Ajout de `created_by` pour lier un worker à un compte utilisateur

- **Nouvelles contraintes** :
  - Workers avec `site_id` : unique par (site_id, email)
  - Workers sans `site_id` : unique par (created_by, email)

- **Politiques RLS mises à jour** :
  - Les utilisateurs peuvent voir leurs workers au niveau du compte
  - Les utilisateurs peuvent créer des workers au niveau du compte

## Fonctionnalités

### Page Équipe (`/team`)
- Ajouter des workers au niveau du compte (sans chantier spécifique)
- Voir tous les workers du compte et des chantiers séparément
- Les workers du compte sont marqués comme "Disponible"

### Page Chantier (`/site/[id]`)
- Sélectionner un worker existant depuis votre équipe
- Ou créer un nouveau worker directement lié au chantier
- Vérification pour éviter les doublons

## Notes importantes

- Les workers existants (avec `site_id`) ne sont pas affectés
- Un worker du compte peut être assigné à plusieurs chantiers (une copie est créée pour chaque chantier)
- Les workers créés depuis la page équipe n'ont pas de `site_id` et sont disponibles pour tous les chantiers

