-- Sécurisation de la table workers : retirer l'accès public en lecture
-- Les workers s'authentifient via /worker/login qui utilise le service_role
-- pour vérifier le code, donc le client n'a jamais besoin de lire directement la table.

DROP POLICY IF EXISTS "Public read access for workers" ON public.workers;

-- Sécuriser aussi les inserts publics (qui ne devraient pas exister)
DROP POLICY IF EXISTS "Public insert access for workers" ON public.workers;
