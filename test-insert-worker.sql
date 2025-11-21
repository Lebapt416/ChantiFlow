-- Test d'insertion d'un worker au niveau du compte
-- Remplacez 'VOTRE_USER_ID' par votre ID utilisateur réel
-- Vous pouvez le récupérer avec: SELECT id FROM auth.users LIMIT 1;

-- Test 1: Vérifier que vous pouvez insérer un worker avec site_id = NULL et created_by
INSERT INTO public.workers (created_by, name, email, role, site_id)
VALUES (
  auth.uid(), -- Votre ID utilisateur
  'Test Worker',
  'test@example.com',
  'Testeur',
  NULL
)
RETURNING id, name, email, created_by, site_id;

-- Si ça fonctionne, supprimez le worker de test:
-- DELETE FROM public.workers WHERE email = 'test@example.com';

