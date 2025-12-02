/**
 * Capitalise la première lettre d'un métier/rôle
 * Exemple: "plombier" -> "Plombier", "maçon" -> "Maçon"
 */
export function capitalizeRole(role: string | null | undefined): string {
  if (!role || !role.trim()) return '';
  
  const trimmed = role.trim();
  // Capitalise la première lettre en préservant les accents
  return trimmed.charAt(0).toUpperCase() + trimmed.slice(1).toLowerCase();
}

/**
 * Capitalise chaque mot d'un métier/rôle composé
 * Exemple: "maçon finisseur" -> "Maçon Finisseur"
 */
export function capitalizeRoleWords(role: string | null | undefined): string {
  if (!role || !role.trim()) return '';
  
  return role
    .trim()
    .split(/\s+/)
    .map(word => capitalizeRole(word))
    .join(' ');
}

