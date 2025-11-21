/**
 * Génère un code d'accès unique de 8 caractères
 * Format: 4 lettres majuscules + 4 chiffres (ex: ABCD1234)
 */
export function generateAccessCode(): string {
  const letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'; // Exclut I, O pour éviter confusion
  const numbers = '0123456789';
  
  let code = '';
  
  // 4 lettres majuscules
  for (let i = 0; i < 4; i++) {
    code += letters.charAt(Math.floor(Math.random() * letters.length));
  }
  
  // 4 chiffres
  for (let i = 0; i < 4; i++) {
    code += numbers.charAt(Math.floor(Math.random() * numbers.length));
  }
  
  return code;
}

/**
 * Valide le format d'un code d'accès
 */
export function isValidAccessCode(code: string): boolean {
  return /^[A-Z]{4}[0-9]{4}$/.test(code);
}

