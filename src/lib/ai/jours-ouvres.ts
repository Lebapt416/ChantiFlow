/**
 * Utilitaires de calcul des jours ouvrés pour le BTP en France.
 *
 * Fonctionnalités :
 *  - Formatage ISO YYYY-MM-DD sans bug de décalage UTC
 *  - Calcul des jours fériés français (fixes + mobiles basés sur Pâques)
 *  - Exclusion des samedis et dimanches (ou inclusion des samedis en option)
 *  - Fonctions de navigation : prochain jour ouvré, ajout de N jours ouvrés,
 *    calcul de date de fin de tâche (premier jour compté)
 */

// ─────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────

/** Options de calcul des jours ouvrés */
export type OptionsJoursOuvres = {
  /**
   * Inclure les samedis comme jours ouvrés.
   * Certains chantiers BTP travaillent le samedi.
   * Défaut : false (samedis exclus).
   */
  includeSaturdays?: boolean;
};

// ─────────────────────────────────────────────────────────────
// FORMATAGE ISO SANS BUG UTC
// ─────────────────────────────────────────────────────────────

/**
 * Formate une date en chaîne ISO YYYY-MM-DD en lisant les composantes locales.
 *
 * ⚠️  Pourquoi ne pas utiliser `date.toISOString().split('T')[0]` ?
 *     `toISOString()` convertit la date en UTC. En France (UTC+1 ou UTC+2),
 *     une date locale à 00h00 est par exemple 22h00 la veille en UTC.
 *     Résultat : la date affichée recule d'un jour — bug silencieux fréquent.
 *     Cette fonction lit directement `getFullYear()`, `getMonth()`, `getDate()`
 *     (composantes locales) et évite ce piège.
 */
export function formatDateISO(date: Date): string {
  const annee = date.getFullYear();
  const mois  = String(date.getMonth() + 1).padStart(2, '0');
  const jour  = String(date.getDate()).padStart(2, '0');
  return `${annee}-${mois}-${jour}`;
}

/**
 * Crée une Date à partir d'une chaîne YYYY-MM-DD sans décalage UTC.
 * `new Date('2026-05-19')` est interprété en UTC (minuit UTC) → bug local.
 * Cette fonction crée la date en heure locale.
 */
export function parseDateISO(iso: string): Date {
  const [annee, mois, jour] = iso.split('-').map(Number);
  return new Date(annee, mois - 1, jour);
}

// ─────────────────────────────────────────────────────────────
// JOURS FÉRIÉS FRANÇAIS
// ─────────────────────────────────────────────────────────────

/**
 * Calcule la date du dimanche de Pâques pour une année donnée.
 *
 * Algorithme : Anonymous Gregorian Algorithm (Meeus/Jones/Butcher).
 * Référence : https://en.wikipedia.org/wiki/Date_of_Easter#Anonymous_Gregorian_algorithm
 */
function dimanchePaques(annee: number): Date {
  const a = annee % 19;
  const b = Math.floor(annee / 100);
  const c = annee % 100;
  const d = Math.floor(b / 4);
  const e = b % 4;
  const f = Math.floor((b + 8) / 25);
  const g = Math.floor((b - f + 1) / 3);
  const h = (19 * a + b - d - g + 15) % 30;
  const i = Math.floor(c / 4);
  const k = c % 4;
  const l = (32 + 2 * e + 2 * i - h - k) % 7;
  const m = Math.floor((a + 11 * h + 22 * l) / 451);
  const moisPaques = Math.floor((h + l - 7 * m + 114) / 31); // 3 = mars, 4 = avril
  const jourPaques = ((h + l - 7 * m + 114) % 31) + 1;
  // Constructeur local (pas UTC) : évite le décalage de fuseau
  return new Date(annee, moisPaques - 1, jourPaques);
}

/**
 * Ajoute un nombre de jours calendaires à une date.
 * Retourne une nouvelle Date sans modifier l'originale.
 */
function ajouterJoursCalendaires(date: Date, jours: number): Date {
  const resultat = new Date(date);
  resultat.setDate(resultat.getDate() + jours);
  return resultat;
}

/**
 * Retourne l'ensemble des jours fériés légaux français pour une année donnée,
 * sous la forme d'un Set<string> au format YYYY-MM-DD.
 *
 * Jours fériés couverts (article L3133-1 du Code du Travail) :
 *  Fixes  : Jour de l'an (1er jan), Fête du Travail (1er mai),
 *            Victoire 1945 (8 mai), Fête Nationale (14 juil),
 *            Assomption (15 août), Toussaint (1er nov),
 *            Armistice 1918 (11 nov), Noël (25 déc)
 *  Mobiles : Lundi de Pâques (Pâques+1), Jeudi de l'Ascension (Pâques+39),
 *            Lundi de Pentecôte (Pâques+50)
 */
export function getFeriesFrancais(annee: number): Set<string> {
  const paques = dimanchePaques(annee);

  const feries: Date[] = [
    // ── Jours fériés fixes ─────────────────────────────────
    new Date(annee, 0,  1),  // Jour de l'an
    new Date(annee, 4,  1),  // Fête du Travail
    new Date(annee, 4,  8),  // Victoire 1945
    new Date(annee, 6, 14),  // Fête Nationale
    new Date(annee, 7, 15),  // Assomption de la Vierge Marie
    new Date(annee, 10, 1),  // Toussaint
    new Date(annee, 10, 11), // Armistice 1918
    new Date(annee, 11, 25), // Noël

    // ── Jours fériés mobiles (basés sur Pâques) ────────────
    ajouterJoursCalendaires(paques,  1), // Lundi de Pâques
    ajouterJoursCalendaires(paques, 39), // Jeudi de l'Ascension
    ajouterJoursCalendaires(paques, 50), // Lundi de Pentecôte
  ];

  return new Set(feries.map(formatDateISO));
}

// ─────────────────────────────────────────────────────────────
// CACHE DES FÉRIÉS PAR ANNÉE
// ─────────────────────────────────────────────────────────────

/**
 * Cache en mémoire des jours fériés par année.
 * Évite de recalculer Pâques à chaque appel sur la même année.
 * Durée de vie : celle du processus Node.js (réinitialisé à chaque déploiement).
 */
const cacheFeriers = new Map<number, Set<string>>();

function getFeriesCached(annee: number): Set<string> {
  if (!cacheFeriers.has(annee)) {
    cacheFeriers.set(annee, getFeriesFrancais(annee));
  }
  return cacheFeriers.get(annee)!;
}

// ─────────────────────────────────────────────────────────────
// FONCTIONS PRINCIPALES
// ─────────────────────────────────────────────────────────────

/**
 * Détermine si une date est un jour ouvré en France.
 *
 * Un jour est non ouvré si :
 *  - c'est un dimanche (toujours)
 *  - c'est un samedi ET `includeSaturdays` est false
 *  - c'est un jour férié légal français
 *
 * @param date            - La date à évaluer
 * @param includeSaturdays - Si true, le samedi compte comme ouvré (défaut : false)
 */
export function estJourOuvre(date: Date, includeSaturdays = false): boolean {
  const jourSemaine = date.getDay(); // 0 = dim, 1 = lun, …, 6 = sam

  // Dimanche : jamais ouvré
  if (jourSemaine === 0) return false;

  // Samedi : ouvré uniquement si l'option est activée
  if (jourSemaine === 6 && !includeSaturdays) return false;

  // Vérifier les jours fériés (cache par année)
  return !getFeriesCached(date.getFullYear()).has(formatDateISO(date));
}

/**
 * Retourne la première date ouvrée ≥ à la date fournie.
 * Si la date est déjà un jour ouvré, elle est retournée telle quelle.
 *
 * Exemple : samedi 17 mai 2026 → lundi 19 mai 2026
 * Exemple : lundi 19 mai 2026 (Whit Monday) → mardi 20 mai 2026
 *
 * @param date            - Date de départ
 * @param includeSaturdays - Si true, le samedi est considéré ouvré
 */
export function prochaineJourOuvre(date: Date, includeSaturdays = false): Date {
  const resultat = new Date(date);
  // Garde-fou : maximum 30 jours d'avance (cas extrême de série de fériés)
  let garde = 0;
  while (!estJourOuvre(resultat, includeSaturdays) && garde < 30) {
    resultat.setDate(resultat.getDate() + 1);
    garde++;
  }
  return resultat;
}

/**
 * Avance d'un nombre précis de jours ouvrés depuis une date de départ.
 *
 * ⚠️  La date de départ N'EST PAS comptée comme premier jour ouvré.
 *     Pour calculer une date de fin de tâche (début = jour 1), utiliser `finDeTache()`.
 *
 * Exemple : ajouterJoursOuvres(vendredi 15 mai, 3)
 *   → samedi, dim ignorés ; lundi=1, mardi=2, mercredi=3 → mercredi 20 mai
 *
 * @param dateDebut       - Date de départ (non incluse dans le décompte)
 * @param joursOuvres     - Nombre de jours ouvrés à avancer (≥ 0)
 * @param includeSaturdays - Si true, le samedi est considéré ouvré
 */
export function ajouterJoursOuvres(
  dateDebut: Date,
  joursOuvres: number,
  includeSaturdays = false,
): Date {
  if (joursOuvres <= 0) return new Date(dateDebut);

  const resultat = new Date(dateDebut);
  let joursComptes = 0;
  // Garde-fou : jours calendaires max = joursOuvres * 3 + 30
  // (couvre les semaines avec plusieurs fériés consécutifs)
  const maxIterations = joursOuvres * 3 + 30;
  let garde = 0;

  while (joursComptes < joursOuvres && garde < maxIterations) {
    resultat.setDate(resultat.getDate() + 1);
    if (estJourOuvre(resultat, includeSaturdays)) {
      joursComptes++;
    }
    garde++;
  }

  return resultat;
}

/**
 * Calcule la date de fin d'une tâche exprimée en jours ouvrés.
 *
 * Convention BTP : le premier jour de travail (dateDebut) est compté comme J1.
 * La durée "1 jour" signifie : début = fin (tâche sur une seule journée).
 *
 * Exemples :
 *   finDeTache(vendredi 15 mai, 1) → vendredi 15 mai   (même jour)
 *   finDeTache(vendredi 15 mai, 3) → mardi 19 mai      (ven=J1, lun=J2, mar=J3)
 *   finDeTache(vendredi 15 mai, 5) → jeudi 21 mai      (ven, lun, mar, mer, jeu)
 *
 * @param dateDebut       - Premier jour de travail (doit être un jour ouvré)
 * @param joursOuvres     - Durée totale en jours ouvrés (≥ 1)
 * @param includeSaturdays - Si true, le samedi est considéré ouvré
 */
export function finDeTache(
  dateDebut: Date,
  joursOuvres: number,
  includeSaturdays = false,
): Date {
  // Le premier jour (dateDebut) compte comme J1 → on ajoute (N-1) jours supplémentaires
  if (joursOuvres <= 1) return new Date(dateDebut);
  return ajouterJoursOuvres(dateDebut, joursOuvres - 1, includeSaturdays);
}
