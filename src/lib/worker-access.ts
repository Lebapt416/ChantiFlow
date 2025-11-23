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
 * Génère un code unique au format 4 chiffres + 4 lettres (ex: 1234ABCD)
 */
export function generateWorkerAccessCodeAlphanumeric(): string {
  // 4 chiffres (0-9)
  const digits = '0123456789';
  let digitPart = '';
  for (let i = 0; i < 4; i++) {
    digitPart += digits.charAt(Math.floor(Math.random() * digits.length));
  }
  
  // 4 lettres (exclut les caractères ambigus: O, I, 0, 1)
  const letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ';
  let letterPart = '';
  for (let i = 0; i < 4; i++) {
    letterPart += letters.charAt(Math.floor(Math.random() * letters.length));
  }
  
  return digitPart + letterPart; // Format: 1234ABCD
}

