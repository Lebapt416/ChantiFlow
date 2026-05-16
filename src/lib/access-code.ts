/**
 * Génère un code d'accès unique de 8 caractères
 * Format: 4 chiffres + 4 lettres majuscules (ex: 1234ABCD)
 */
export function generateAccessCode(): string {
  const numbers = '0123456789';
  const letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'; // Exclut I, O pour éviter confusion

  let code = '';

  // 4 chiffres D'ABORD
  for (let i = 0; i < 4; i++) {
    code += numbers.charAt(Math.floor(Math.random() * numbers.length));
  }

  // Puis 4 lettres majuscules
  for (let i = 0; i < 4; i++) {
    code += letters.charAt(Math.floor(Math.random() * letters.length));
  }

  return code;
}

/**
 * Valide le format d'un code d'accès
 */
export function isValidAccessCode(code: string): boolean {
  return /^[0-9]{4}[A-Z]{4}$/.test(code);
}
