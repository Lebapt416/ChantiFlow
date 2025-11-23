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
 * Format strict: 4 chiffres suivis de 4 lettres majuscules
 */
export function generateWorkerAccessCodeAlphanumeric(): string {
  // 4 chiffres (0-9) - toujours en premier
  const digits = '0123456789';
  let digitPart = '';
  for (let i = 0; i < 4; i++) {
    digitPart += digits.charAt(Math.floor(Math.random() * digits.length));
  }
  
  // 4 lettres majuscules (exclut les caractères ambigus: O, I)
  // On exclut O et I pour éviter la confusion avec 0 et 1
  const letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ';
  let letterPart = '';
  for (let i = 0; i < 4; i++) {
    letterPart += letters.charAt(Math.floor(Math.random() * letters.length));
  }
  
  // Format strict: 4 chiffres + 4 lettres (ex: 1234ABCD)
  const code = digitPart + letterPart;
  
  // Vérification de sécurité: s'assurer que le code est bien au format attendu
  if (!/^[0-9]{4}[A-Z]{4}$/.test(code)) {
    console.error('Erreur: Code généré ne respecte pas le format attendu:', code);
    // Régénérer si nécessaire
    return generateWorkerAccessCodeAlphanumeric();
  }
  
  return code;
}

