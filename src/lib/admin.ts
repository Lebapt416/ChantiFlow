/**
 * Utilitaires pour la gestion des administrateurs
 */

/**
 * Vérifie si un utilisateur est administrateur
 * Les admins sont définis via la variable d'environnement ADMIN_EMAILS
 * (liste d'emails séparés par des virgules)
 * 
 * En mode développement (NODE_ENV !== 'production'), si ADMIN_EMAILS n'est pas défini,
 * tous les utilisateurs sont considérés comme admins pour faciliter les tests.
 */
export function isAdmin(email: string | null | undefined): boolean {
  if (!email) return false;
  
  const adminEmails = process.env.ADMIN_EMAILS;
  
  // En développement, si ADMIN_EMAILS n'est pas défini, permettre l'accès admin à tous
  // pour faciliter les tests (sécurité : uniquement en dev)
  if (!adminEmails && process.env.NODE_ENV !== 'production') {
    console.warn('[Admin] Mode développement : accès admin activé pour tous (définissez ADMIN_EMAILS en production)');
    return true;
  }
  
  if (!adminEmails) return false;
  
  const emails = adminEmails.split(',').map(e => e.trim().toLowerCase());
  return emails.includes(email.toLowerCase());
}

