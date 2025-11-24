/**
 * Règles de travail par métier pour l'optimisation du planning
 * Définit les horaires, conditions météo et contraintes spécifiques
 */

export type WorkRule = {
  role: string;
  workingHours: {
    start: string; // Format "HH:mm"
    end: string;
    breakDuration: number; // en minutes
  };
  weatherConstraints: {
    avoidRain: boolean; // Éviter la pluie
    avoidExtremeCold: boolean; // Éviter le froid extrême (< 0°C)
    avoidExtremeHeat: boolean; // Éviter la chaleur extrême (> 35°C)
    minTemperature?: number; // Température minimale acceptable
    maxTemperature?: number; // Température maximale acceptable
  };
  priority: 'high' | 'medium' | 'low'; // Priorité pour l'optimisation
};

export const WORK_RULES: Record<string, WorkRule> = {
  // Maçon / Carreleur
  maçon: {
    role: 'maçon',
    workingHours: {
      start: '07:00',
      end: '17:00',
      breakDuration: 60, // 1h de pause déjeuner
    },
    weatherConstraints: {
      avoidRain: true,
      avoidExtremeCold: true,
      avoidExtremeHeat: false,
      minTemperature: 5,
      maxTemperature: 35,
    },
    priority: 'high',
  },
  carreleur: {
    role: 'carreleur',
    workingHours: {
      start: '07:00',
      end: '17:00',
      breakDuration: 60,
    },
    weatherConstraints: {
      avoidRain: true,
      avoidExtremeCold: true,
      avoidExtremeHeat: false,
      minTemperature: 5,
    },
    priority: 'high',
  },
  // Charpentier / Menuisier
  charpentier: {
    role: 'charpentier',
    workingHours: {
      start: '07:00',
      end: '17:00',
      breakDuration: 60,
    },
    weatherConstraints: {
      avoidRain: true,
      avoidExtremeCold: true,
      avoidExtremeHeat: false,
      minTemperature: 0,
    },
    priority: 'high',
  },
  menuisier: {
    role: 'menuisier',
    workingHours: {
      start: '08:00',
      end: '18:00',
      breakDuration: 60,
    },
    weatherConstraints: {
      avoidRain: false, // Travail souvent en intérieur
      avoidExtremeCold: false,
      avoidExtremeHeat: false,
    },
    priority: 'medium',
  },
  // Électricien / Plombier
  électricien: {
    role: 'électricien',
    workingHours: {
      start: '08:00',
      end: '18:00',
      breakDuration: 60,
    },
    weatherConstraints: {
      avoidRain: true, // Travail extérieur possible
      avoidExtremeCold: false,
      avoidExtremeHeat: false,
    },
    priority: 'high',
  },
  plombier: {
    role: 'plombier',
    workingHours: {
      start: '08:00',
      end: '18:00',
      breakDuration: 60,
    },
    weatherConstraints: {
      avoidRain: false, // Travail souvent en intérieur
      avoidExtremeCold: false,
      avoidExtremeHeat: false,
    },
    priority: 'medium',
  },
  // Peintre
  peintre: {
    role: 'peintre',
    workingHours: {
      start: '08:00',
      end: '18:00',
      breakDuration: 60,
    },
    weatherConstraints: {
      avoidRain: true,
      avoidExtremeCold: true,
      avoidExtremeHeat: false,
      minTemperature: 10,
      maxTemperature: 30,
    },
    priority: 'high',
  },
  // Terrassier
  terrassier: {
    role: 'terrassier',
    workingHours: {
      start: '07:00',
      end: '17:00',
      breakDuration: 60,
    },
    weatherConstraints: {
      avoidRain: true, // Sol mouillé = problème
      avoidExtremeCold: true,
      avoidExtremeHeat: false,
      minTemperature: 0,
    },
    priority: 'high',
  },
  // Couvreur
  couvreur: {
    role: 'couvreur',
    workingHours: {
      start: '07:00',
      end: '17:00',
      breakDuration: 60,
    },
    weatherConstraints: {
      avoidRain: true,
      avoidExtremeCold: true,
      avoidExtremeHeat: false,
      minTemperature: 5,
      maxTemperature: 30,
    },
    priority: 'high',
  },
  // Par défaut pour les autres métiers
  default: {
    role: 'default',
    workingHours: {
      start: '08:00',
      end: '17:00',
      breakDuration: 60,
    },
    weatherConstraints: {
      avoidRain: false,
      avoidExtremeCold: false,
      avoidExtremeHeat: false,
    },
    priority: 'low',
  },
};

/**
 * Récupère les règles de travail pour un métier donné
 */
export function getWorkRule(role: string | null | undefined): WorkRule {
  if (!role) return WORK_RULES.default;

  const normalizedRole = role.toLowerCase().trim();
  
  // Recherche exacte
  if (WORK_RULES[normalizedRole]) {
    return WORK_RULES[normalizedRole];
  }

  // Recherche par mot-clé
  for (const [key, rule] of Object.entries(WORK_RULES)) {
    if (key !== 'default' && normalizedRole.includes(key)) {
      return rule;
    }
    if (normalizedRole.includes(rule.role)) {
      return rule;
    }
  }

  return WORK_RULES.default;
}

/**
 * Vérifie si les conditions météo sont favorables pour un métier
 */
export function isWeatherFavorable(
  rule: WorkRule,
  temperature: number,
  precipitation: number,
): { favorable: boolean; reason?: string } {
  const { weatherConstraints } = rule;

  if (weatherConstraints.avoidRain && precipitation > 0.5) {
    return {
      favorable: false,
      reason: `Pluie prévue (${Math.round(precipitation * 10)}%) - à éviter pour ${rule.role}`,
    };
  }

  if (weatherConstraints.avoidExtremeCold && temperature < 0) {
    return {
      favorable: false,
      reason: `Température trop basse (${temperature}°C) - à éviter pour ${rule.role}`,
    };
  }

  if (weatherConstraints.avoidExtremeHeat && temperature > 35) {
    return {
      favorable: false,
      reason: `Température trop élevée (${temperature}°C) - à éviter pour ${rule.role}`,
    };
  }

  if (weatherConstraints.minTemperature && temperature < weatherConstraints.minTemperature) {
    return {
      favorable: false,
      reason: `Température minimale non atteinte (${temperature}°C < ${weatherConstraints.minTemperature}°C)`,
    };
  }

  if (weatherConstraints.maxTemperature && temperature > weatherConstraints.maxTemperature) {
    return {
      favorable: false,
      reason: `Température maximale dépassée (${temperature}°C > ${weatherConstraints.maxTemperature}°C)`,
    };
  }

  return { favorable: true };
}

/**
 * Calcule les heures de travail effectives en tenant compte des pauses
 */
export function getEffectiveWorkingHours(rule: WorkRule): number {
  const start = rule.workingHours.start.split(':').map(Number);
  const end = rule.workingHours.end.split(':').map(Number);
  
  const startMinutes = start[0] * 60 + start[1];
  const endMinutes = end[0] * 60 + end[1];
  
  const totalMinutes = endMinutes - startMinutes - rule.workingHours.breakDuration;
  return totalMinutes / 60;
}

