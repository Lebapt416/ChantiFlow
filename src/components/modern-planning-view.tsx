'use client';

import { CalendarDays, CheckCircle2, AlertTriangle, Clock } from 'lucide-react';

type PlanningPhase = {
  id: string;
  name: string;
  status: 'completed' | 'in_progress' | 'pending' | 'delayed';
  progress: number; // 0-100
  assignedWorker?: {
    id: string;
    name: string;
    avatar?: string;
  };
};

type Props = {
  siteName: string;
  phases: PlanningPhase[];
  isAheadOfSchedule?: boolean;
  showAIBadge?: boolean;
};

/**
 * Normalise un nom de métier pour la recherche
 */
function normalizeRole(role: string | null | undefined): string {
  if (!role) return '';
  return role
    .toLowerCase()
    .trim()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '') // Supprime les accents
    .replace(/[^a-z0-9\s]/g, ' ') // Remplace les caractères spéciaux par des espaces
    .replace(/\s+/g, ' ') // Remplace les espaces multiples par un seul espace
    .trim();
}

/**
 * Détermine la catégorie d'un métier et retourne sa couleur selon le code couleur :
 * - Gros Œuvre = Vert (emerald)
 * - Second Œuvre (Élec/Plomberie) = Bleu/Cyan
 * - Finitions (Peinture/Sols) = Magenta/Vert
 * - Encadrement/Études = Violet
 * - TP / Engins = Orange
 */
function getRoleColor(roleName: string | null | undefined): {
  labelColor: string;
  bgColor: string;
  borderColor: string;
  barColor: string; // Couleur de la barre verticale
  barNeonColor: string; // Effet néon pour la barre
  category: string; // Nom de la catégorie pour la légende
} {
  const normalized = normalizeRole(roleName);
  if (!normalized) {
    // Fallback par défaut
    return { 
      labelColor: 'text-zinc-400', 
      bgColor: 'bg-zinc-800/80', 
      borderColor: 'border-zinc-500',
      barColor: 'bg-zinc-500',
      barNeonColor: 'shadow-[0_0_8px_rgba(113,113,122,0.4)]',
      category: 'Autre'
    };
  }

  // Recherche directe pour les métiers les plus courants (optimisation)
  const directMatches: Record<string, { labelColor: string; bgColor: string; borderColor: string; barColor: string; barNeonColor: string; category: string }> = {
    // Gros Œuvre
    'macon': { labelColor: 'text-emerald-300', bgColor: 'bg-emerald-950/30', borderColor: 'border-zinc-700', barColor: 'bg-emerald-500', barNeonColor: 'shadow-[0_0_8px_rgba(16,185,129,0.6)]', category: 'Gros Œuvre' },
    'plombier': { labelColor: 'text-cyan-300', bgColor: 'bg-cyan-950/30', borderColor: 'border-zinc-700', barColor: 'bg-cyan-500', barNeonColor: 'shadow-[0_0_8px_rgba(6,182,212,0.6)]', category: 'Fluides & Tech' },
    'architecte': { labelColor: 'text-violet-300', bgColor: 'bg-violet-950/30', borderColor: 'border-zinc-700', barColor: 'bg-violet-500', barNeonColor: 'shadow-[0_0_8px_rgba(139,92,246,0.6)]', category: 'Études / Gestion' },
    'salinier': { labelColor: 'text-emerald-300', bgColor: 'bg-emerald-950/30', borderColor: 'border-zinc-700', barColor: 'bg-emerald-500', barNeonColor: 'shadow-[0_0_8px_rgba(16,185,129,0.6)]', category: 'Gros Œuvre' },
    'electricien': { labelColor: 'text-blue-300', bgColor: 'bg-blue-950/30', borderColor: 'border-zinc-700', barColor: 'bg-blue-500', barNeonColor: 'shadow-[0_0_8px_rgba(59,130,246,0.6)]', category: 'Fluides & Tech' },
    'peintre': { labelColor: 'text-fuchsia-300', bgColor: 'bg-fuchsia-950/30', borderColor: 'border-zinc-700', barColor: 'bg-fuchsia-500', barNeonColor: 'shadow-[0_0_8px_rgba(217,70,239,0.6)]', category: 'Finitions' },
    'grutier': { labelColor: 'text-orange-300', bgColor: 'bg-orange-950/30', borderColor: 'border-zinc-700', barColor: 'bg-orange-500', barNeonColor: 'shadow-[0_0_8px_rgba(249,115,22,0.6)]', category: 'TP / Engins' },
  };

  // Vérifier d'abord les correspondances directes
  if (directMatches[normalized]) {
    return directMatches[normalized];
  }

  // Catégorie 1: GROS ŒUVRE & STRUCTURE = Gris/Noir
  const grosOeuvreKeywords = [
    'macon', 'maçon', 'maçonnerie', 'maconnerie', 'macon finisseur', 'macon vrd', 'macon coffreur',
    'coffreur', 'coffreur bancheur', 'coffreur boiseur', 'ferrailleur', 'charpentier bois', 'charpentier metallique',
    'charpentier', 'charpente', 'couvreur', 'couvreur zingueur', 'couvreur ardoisier', 'etancheur', 'bardeur',
    'facadier', 'facadier iteiste', 'isolation thermique exterieur', 'ite', 'tailleur de pierre',
    'monteur structures metalliques', 'monteur structures bois', 'demolisseur', 'scieur de beton',
    'carotteur', 'soudeur btp', 'enduiseur de facade', 'jointeur facade', 'conducteur de travaux gros oeuvre',
    'gros oeuvre', 'gros-oeuvre', 'structure', 'beton', 'beton arme', 'ferraillage', 'coffrage', 'banche',
    'terrassier', 'terrassement', 'vr', 'voirie', 'reseau divers', 'canalisateur', 'constructeur de routes',
    'tireur denrobe', 'poseur de bordures', 'pavic', 'ouvrier vrd', 'mineur boiseur', 'puisatier', 'sondeur',
    'geotechnique', 'foreur', 'agent entretien voirie', 'eagueur grimpeur', 'paysagiste', 'espaces verts',
    'monteur reseaux electriques', 'technicien fibre optique', 'monteur catenaire', 'scaphandrier',
    'cordiste', 'signaleur de chantier', 'grutier', 'grue', 'conducteur de pelle', 'conducteur de chargeuse',
    'conducteur de tractopelle', 'conducteur de bouteur', 'bulldozer', 'conducteur de niveleuse',
    'conducteur de decapeuse', 'scraper', 'conducteur de compacteur', 'conducteur de finisseur',
    'conducteur de tombereau', 'dumper', 'conducteur de camion benne', 'conducteur de camion toupie',
    'conducteur de camion pompe', 'pompiste beton', 'conducteur de nacelle', 'pemp', 'conducteur de chariot',
    'magasinier de chantier', 'responsable logistique chantier', 'tunnelier', 'pilote de tunnelier',
    'constructeur ouvrages dart', 'poseur de voies ferrees', 'soudeur de rails', 'technicien signalisation ferroviaire',
    'agent de maintenance ouvrages dart', 'ingenieur ponts et chaussees', 'genie civil', 'ouvrages dart',
    'salinier' // Métier spécialisé (travaux salins)
  ];

  // Catégorie 2: SECOND ŒUVRE (Élec/Plomberie) = Bleu/Cyan
  const secondOeuvreKeywords = [
    'electricien', 'electricien batiment', 'electricien tertiaire', 'electricite', 'elec',
    'plombier', 'plomberie', 'plombier chauffagiste', 'chauffagiste', 'chauffage', 'cvc',
    'chauffage ventilation clim', 'frigoriste', 'installateur thermique', 'installateur pompe a chaleur',
    'installateur panneaux photovoltaiques', 'technicien maintenance cvc', 'domoticien', 'auditeur energetique',
    'monteur installation thermique', 'tuyauteur industriel', 'calorifugeur', 'technicien traitement des eaux',
    'second oeuvre', 'second-oeuvre'
  ];

  // Catégorie 3: FINITIONS (Peinture/Sols) = Vert/Orange
  const finitionsKeywords = [
    'peintre', 'peintre en batiment', 'peintre decorateur', 'peintre enduiseur', 'peintre solier',
    'peinture', 'plaquiste', 'jointeur placo', 'staffeur', 'ornemaniste', 'menuisier poseur',
    'menuisier aluminium', 'menuisier pvc', 'menuisier bois', 'menuisier dagencement', 'menuiserie',
    'serrurier metallier', 'ferronnier dart', 'vitrier', 'miroitier', 'carreleur', 'mosaiste',
    'carrelage', 'solier', 'moquettiste', 'parqueteur', 'agenceur de cuisine', 'agenceur de bain',
    'poseur de cloisons amovibles', 'poseur de faux plafonds', 'cheministe', 'installateur de cheminees',
    'fumiste', 'conduits de fumee', 'installateur de stores', 'installateur de volets', 'finition',
    'finitions', 'peintre ravaleur', 'applicateur de resine', 'applicateur de beton cire', 'hydrogommeur',
    'sableur', 'pisciniste', 'installateur de clôtures', 'installateur de portails', 'installateur de verandas'
  ];

  // Catégorie 4: ENCADREMENT & ÉTUDES = Violet
  const encadrementKeywords = [
    'architecte', 'architecte dplg', 'architecte de', 'architecte dinterieur', 'architecture',
    'dessinateur projeteur', 'dessinateur btp', 'autocad', 'revit', 'ingenieur structure',
    'ingenieur beton arme', 'ingenieur fluides', 'ingenieur electricite', 'ingenieur geotechnicien',
    'ingenieur acoustique', 'ingenieur travaux', 'ingenieur methodes', 'economiste construction',
    'geometre topographe', 'geometre expert', 'bim manager', 'bim coordinateur', 'bim modeleur',
    'metreur verificateur', 'charge daffaires btp', 'ingenieur commercial btp', 'directeur de travaux',
    'conducteur de travaux principal', 'conducteur de travaux', 'aide conducteur de travaux',
    'chef de chantier', 'chef de chantier gros oeuvre', 'chef de chantier tp', 'chef dequipe btp',
    'chef dequipe maconnerie', 'chef dequipe electricite', 'opc', 'ordonnancement pilotage coordination',
    'maitre doeuvre execution', 'moe', 'assistant a maitrise douvrage', 'amo', 'coordonnateur sps',
    'securite protection sante', 'controleur technique construction', 'responsable qse',
    'qualite securite environnement', 'secretaire technique btp', 'comptable btp', 'gestionnaire de paie btp',
    'acheteur btp', 'responsable materiel', 'dispatcheur', 'agent de bascule', 'commercial materiaux',
    'promoteur immobilier', 'responsable programmes immobiliers', 'charge detudes de prix',
    'technicien bureau detudes structure', 'technicien bureau detudes fluides', 'technicien bureau detudes elec',
    'projeteur beton arme', 'projeteur charpente metallique', 'coordinateur ssi', 'systeme securite incendie',
    'expert en batiment', 'assurances', 'sinistres', 'technicien geometre', 'traceur de chantier',
    'encadrement', 'etudes', 'conception', 'ingenierie'
  ];

  // Vérifier dans l'ordre des catégories
  const checkKeywords = (keywords: string[]) => {
    // D'abord, recherche exacte
    if (keywords.includes(normalized)) {
      return true;
    }
    // Ensuite, recherche par inclusion (le nom contient le keyword)
    return keywords.some(keyword => {
      // Recherche exacte du mot entier
      if (normalized === keyword) return true;
      // Recherche par inclusion (le nom normalisé contient le keyword)
      if (normalized.includes(keyword)) return true;
      // Recherche inversée (le keyword contient le nom normalisé) - seulement si le nom est assez long
      if (normalized.length >= 3 && keyword.includes(normalized)) return true;
      // Recherche par mots individuels (pour gérer "maçon finisseur" vs "maçon")
      const normalizedWords = normalized.split(' ');
      const keywordWords = keyword.split(' ');
      if (normalizedWords.some(word => keywordWords.includes(word) && word.length >= 3)) return true;
      return false;
    });
  };

  // Catégorie 1: Gros Œuvre = Vert (emerald)
  if (checkKeywords(grosOeuvreKeywords)) {
    return { 
      labelColor: 'text-emerald-300', 
      bgColor: 'bg-emerald-950/30', 
      borderColor: 'border-zinc-700',
      barColor: 'bg-emerald-500',
      barNeonColor: 'shadow-[0_0_8px_rgba(16,185,129,0.6)]',
      category: 'Gros Œuvre'
    };
  }

  // Catégorie 2: Second Œuvre (Élec/Plomberie) = Bleu/Cyan
  if (checkKeywords(secondOeuvreKeywords)) {
    // Électricité = Bleu
    if (normalized.includes('electric') || normalized.includes('elec')) {
      return { 
        labelColor: 'text-blue-300', 
        bgColor: 'bg-blue-950/30', 
        borderColor: 'border-zinc-700',
        barColor: 'bg-blue-500',
        barNeonColor: 'shadow-[0_0_8px_rgba(59,130,246,0.6)]',
        category: 'Fluides & Tech'
      };
    }
    // Plomberie/CVC = Cyan
    return { 
      labelColor: 'text-cyan-300', 
      bgColor: 'bg-cyan-950/30', 
      borderColor: 'border-zinc-700',
      barColor: 'bg-cyan-500',
      barNeonColor: 'shadow-[0_0_8px_rgba(6,182,212,0.6)]',
      category: 'Fluides & Tech'
    };
  }

  // Catégorie 3: Finitions = Magenta/Vert
  if (checkKeywords(finitionsKeywords)) {
    // Peinture = Magenta/Fuchsia
    if (normalized.includes('peintre') || normalized.includes('peinture')) {
      return { 
        labelColor: 'text-fuchsia-300', 
        bgColor: 'bg-fuchsia-950/30', 
        borderColor: 'border-zinc-700',
        barColor: 'bg-fuchsia-500',
        barNeonColor: 'shadow-[0_0_8px_rgba(217,70,239,0.6)]',
        category: 'Finitions'
      };
    }
    // Sols/Menuiserie = Vert (emerald)
    return { 
      labelColor: 'text-emerald-300', 
      bgColor: 'bg-emerald-950/30', 
      borderColor: 'border-zinc-700',
      barColor: 'bg-emerald-500',
      barNeonColor: 'shadow-[0_0_8px_rgba(16,185,129,0.6)]',
      category: 'Finitions'
    };
  }

  // Catégorie 4: Encadrement/Études = Violet
  if (checkKeywords(encadrementKeywords)) {
    return { 
      labelColor: 'text-violet-300', 
      bgColor: 'bg-violet-950/30', 
      borderColor: 'border-zinc-700',
      barColor: 'bg-violet-500',
      barNeonColor: 'shadow-[0_0_8px_rgba(139,92,246,0.6)]',
      category: 'Études / Gestion'
    };
  }

  // TP / Engins = Orange
  const tpEnginsKeywords = [
    'grutier', 'grue', 'conducteur de pelle', 'conducteur de chargeuse', 'conducteur de tractopelle',
    'conducteur de bouteur', 'bulldozer', 'conducteur de niveleuse', 'conducteur de decapeuse',
    'scraper', 'conducteur de compacteur', 'conducteur de finisseur', 'conducteur de tombereau',
    'dumper', 'conducteur de camion benne', 'conducteur de camion toupie', 'conducteur de camion pompe',
    'pompiste beton', 'conducteur de nacelle', 'pemp', 'conducteur de chariot', 'terrassier',
    'terrassement', 'tp', 'travaux publics', 'engins'
  ];
  if (checkKeywords(tpEnginsKeywords)) {
    return { 
      labelColor: 'text-orange-300', 
      bgColor: 'bg-orange-950/30', 
      borderColor: 'border-zinc-700',
      barColor: 'bg-orange-500',
      barNeonColor: 'shadow-[0_0_8px_rgba(249,115,22,0.6)]',
      category: 'TP / Engins'
    };
  }

  // Métiers d'Art & Patrimoine = Couleur spéciale (teal)
  const metiersArtKeywords = [
    'restaurateur monuments historiques', 'mosaiste dart', 'vitrailliste', 'sculpteur sur pierre',
    'doreur a la feuille', 'peintre en decor', 'stucateur', 'charpentier de marine', 'chaumier',
    'lauzier', 'metiers dart', 'patrimoine'
  ];
  if (checkKeywords(metiersArtKeywords)) {
    return { 
      labelColor: 'text-teal-300', 
      bgColor: 'bg-teal-950/30', 
      borderColor: 'border-zinc-700',
      barColor: 'bg-teal-500',
      barNeonColor: 'shadow-[0_0_8px_rgba(20,184,166,0.6)]',
      category: 'Finitions'
    };
  }

  // Maintenance & Services = Couleur spéciale (rose)
  const maintenanceKeywords = [
    'agent maintenance batiment', 'ramoneur', 'deratiseur', 'desinsectiseur', 'technicien ascensoriste',
    'monteur dascenseurs', 'technicien portes automatiques', 'installateur alarmes', 'video surveillance',
    'serrurier depanneur', 'vitrier depanneur', 'maintenance', 'services generaux'
  ];
  if (checkKeywords(maintenanceKeywords)) {
    return { 
      labelColor: 'text-rose-300', 
      bgColor: 'bg-rose-950/30', 
      borderColor: 'border-zinc-700',
      barColor: 'bg-rose-500',
      barNeonColor: 'shadow-[0_0_8px_rgba(244,63,94,0.6)]',
      category: 'Autre'
    };
  }

  // Spécialités diverses = Couleur spéciale (indigo)
  const specialitesKeywords = [
    'diagnostiqueur immobilier', 'amiante', 'plomb', 'dpe', 'desamianteur', 'operateur desamiantage',
    'encadrant technique desamiantage', 'poseur signalisation routiere', 'eclairagiste public',
    'specialites', 'diverses'
  ];
  if (checkKeywords(specialitesKeywords)) {
    return { 
      labelColor: 'text-indigo-300', 
      bgColor: 'bg-indigo-950/30', 
      borderColor: 'border-zinc-700',
      barColor: 'bg-indigo-500',
      barNeonColor: 'shadow-[0_0_8px_rgba(99,102,241,0.6)]',
      category: 'Autre'
    };
  }

  // Fallback : générer une couleur basée sur le hash du nom pour les métiers non reconnus
  // Cela garantit qu'un même métier aura toujours la même couleur
  const hash = normalized.split('').reduce((acc, char) => {
    return ((acc << 5) - acc) + char.charCodeAt(0);
  }, 0);
  
  const fallbackColors = [
    { labelColor: 'text-zinc-400', bgColor: 'bg-zinc-950/30', borderColor: 'border-zinc-700', barColor: 'bg-zinc-500', barNeonColor: 'shadow-[0_0_8px_rgba(113,113,122,0.4)]', category: 'Autre' },
    { labelColor: 'text-slate-400', bgColor: 'bg-slate-950/30', borderColor: 'border-zinc-700', barColor: 'bg-slate-500', barNeonColor: 'shadow-[0_0_8px_rgba(100,116,139,0.4)]', category: 'Autre' },
    { labelColor: 'text-gray-400', bgColor: 'bg-gray-950/30', borderColor: 'border-zinc-700', barColor: 'bg-gray-500', barNeonColor: 'shadow-[0_0_8px_rgba(107,114,128,0.4)]', category: 'Autre' },
  ];
  
  return fallbackColors[Math.abs(hash) % fallbackColors.length];
}

export function ModernPlanningView({ siteName, phases, isAheadOfSchedule = false, showAIBadge = false }: Props) {
  // Calculer la progression globale du projet
  const overallProgress = phases.length > 0
    ? Math.round(phases.reduce((sum, phase) => sum + phase.progress, 0) / phases.length)
    : 0;

  // Fonction pour obtenir les styles de statut selon l'image
  const getStatusConfig = (status: PlanningPhase['status']) => {
    switch (status) {
      case 'completed':
        return {
          statusText: '✓ Terminé',
          statusIcon: CheckCircle2,
        };
      case 'in_progress':
        return {
          statusText: 'En cours',
          statusIcon: Clock,
        };
      case 'delayed':
        return {
          statusText: '▲ Retard',
          statusIcon: AlertTriangle,
        };
      case 'pending':
      default:
        return {
          statusText: 'À venir',
          statusIcon: Clock,
        };
    }
  };

  return (
    <div className="relative overflow-hidden rounded-3xl border border-zinc-800/50 bg-zinc-900 p-6 shadow-2xl">
      {/* Header avec icône et titre */}
      <div className="mb-6 flex items-start gap-4">
        {/* Icône calendrier */}
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-emerald-600">
          <CalendarDays className="h-6 w-6 text-white" />
        </div>
        
        <div className="flex-1">
          <h2 className="mb-2 text-2xl font-bold text-white">
            Planning Intelligent
          </h2>
          <p className="text-sm text-zinc-300">
            Glissez-déposez vos tâches. Si un retard survient, l&apos;IA recalcule automatiquement la fin du chantier.
          </p>
        </div>
      </div>

      {/* Barre de progression globale */}
      <div className="mb-6">
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
          <div
            className="h-full rounded-full bg-emerald-500 transition-all duration-500"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
      </div>

      {/* Cartes de phases horizontales */}
      <div className="mb-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {phases.map((phase) => {
          const statusConfig = getStatusConfig(phase.status);
          const StatusIcon = statusConfig.statusIcon;
          
          // Obtenir les couleurs basées sur le métier (nom de la phase)
          const roleColors = getRoleColor(phase.name);

          // Déterminer le style du badge selon le statut
          const getBadgeStyle = () => {
            switch (phase.status) {
              case 'completed':
                return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
              case 'delayed':
                return 'bg-red-500/20 text-red-400 border-red-500/30';
              case 'in_progress':
                return 'bg-zinc-500/20 text-zinc-300 border-zinc-500/30';
              case 'pending':
              default:
                return 'bg-zinc-500/20 text-zinc-300 border-zinc-500/30';
            }
          };

          return (
            <div
              key={phase.id}
              className="relative overflow-hidden rounded-xl border border-zinc-700 bg-zinc-900/80 p-4"
            >
              {/* Barre verticale colorée à gauche avec effet néon */}
              <div className={`absolute left-0 top-0 bottom-0 w-1 ${roleColors.barColor} ${roleColors.barNeonColor}`} />

              {/* Contenu de la carte */}
              <div className="pl-3">
                {/* Titre coloré */}
                <div className={`mb-3 text-sm font-semibold ${roleColors.labelColor}`}>
                  {phase.name}
                </div>

                {/* Badge de statut avec icône */}
                <div className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${getBadgeStyle()}`}>
                  <StatusIcon className="h-3 w-3 shrink-0" />
                  <span>{statusConfig.statusText}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Légende des catégories */}
      <div className="mt-6 flex flex-wrap items-center gap-4 border-t border-zinc-800 pt-4">
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-emerald-500" />
          <span className="text-xs text-zinc-300">Gros Œuvre</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-blue-500" />
          <span className="text-xs text-zinc-300">Fluides & Tech</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-fuchsia-500" />
          <span className="text-xs text-zinc-300">Finitions</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-orange-500" />
          <span className="text-xs text-zinc-300">TP / Engins</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-violet-500" />
          <span className="text-xs text-zinc-300">Études / Gestion</span>
        </div>
      </div>

      {/* Message de statut IA en bas */}
      {showAIBadge && (
        <div className="flex items-center gap-2 text-sm text-white">
          <div className="h-2 w-2 rounded-full bg-emerald-500" />
          <span>IA recalcule automatiquement la fin du chantier</span>
        </div>
      )}
    </div>
  );
}

