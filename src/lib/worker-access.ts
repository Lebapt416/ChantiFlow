/**
 * Gestion des codes d'accès uniques pour les workers
 */

/**
 * Génère un code unique de 6 chiffres pour l'accès worker
 */
export function generateWorkerAccessCode(): string {
  // Génère un code à 6 chiffres (000000-999999)
  return Math.floor(100000 + Math.random() * 900000).toString();
}

/**
 * Génère un code unique alphanumérique de 8 caractères
 */
export function generateWorkerAccessCodeAlphanumeric(): string {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // Exclut les caractères ambigus (0, O, I, 1)
  let code = '';
  for (let i = 0; i < 8; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return code;
}

