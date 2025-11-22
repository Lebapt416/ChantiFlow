'use client';

import { useEffect, useState } from 'react';
import { Download } from 'lucide-react';

export function InstallPwaButton() {
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null);
  const [isInstalled, setIsInstalled] = useState(false);

  useEffect(() => {
    // Vérifier si l'app est déjà installée
    if (window.matchMedia('(display-mode: standalone)').matches) {
      setIsInstalled(true);
      return;
    }

    // Écouter l'événement beforeinstallprompt
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);

    // Vérifier si l'app est installée après le chargement
    const checkInstalled = () => {
      if (window.matchMedia('(display-mode: standalone)').matches) {
        setIsInstalled(true);
      }
    };

    checkInstalled();
    window.addEventListener('appinstalled', () => {
      setIsInstalled(true);
      setDeferredPrompt(null);
    });

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    };
  }, []);

  const handleInstallClick = async () => {
    if (!deferredPrompt) {
      // Si le prompt n'est pas disponible, donner des instructions
      if (window.matchMedia('(display-mode: standalone)').matches) {
        alert('L\'application est déjà installée !');
      } else {
        // Instructions pour différents navigateurs
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
        const isAndroid = /Android/.test(navigator.userAgent);
        
        if (isIOS) {
          alert('Pour installer sur iOS :\n1. Appuyez sur le bouton Partager\n2. Sélectionnez "Sur l\'écran d\'accueil"');
        } else if (isAndroid) {
          alert('Pour installer sur Android :\n1. Utilisez le menu du navigateur (⋮)\n2. Sélectionnez "Installer l\'application" ou "Ajouter à l\'écran d\'accueil"');
        } else {
          alert('Pour installer sur PC :\n1. Utilisez le menu du navigateur\n2. Cherchez "Installer ChantiFlow" ou l\'icône d\'installation dans la barre d\'adresse');
        }
      }
      return;
    }

    try {
      // Afficher le prompt d'installation
      deferredPrompt.prompt();

      // Attendre la réponse de l'utilisateur
      const { outcome } = await deferredPrompt.userChoice;

      if (outcome === 'accepted') {
        console.log('Installation acceptée');
        setIsInstalled(true);
        alert('Installation réussie ! L\'application va s\'ouvrir.');
      } else {
        console.log('Installation refusée');
      }

      // Réinitialiser le prompt
      setDeferredPrompt(null);
    } catch (error) {
      console.error('Erreur lors de l\'installation:', error);
      alert('Erreur lors de l\'installation. Veuillez réessayer.');
    }
  };

  // Ne pas afficher le bouton si l'app est déjà installée
  if (isInstalled) {
    return null;
  }

  return (
    <button
      onClick={handleInstallClick}
      className="group/item relative flex items-center justify-center w-14 h-14 rounded-xl transition-all duration-200 text-white hover:text-white"
      title="Installer l'application"
    >
      <span className="absolute rounded-xl transition-all duration-200 top-0 bottom-0 left-2 right-0 bg-black/50 group-hover/item:bg-black/70 group-hover/item:left-3"></span>
      <span className="relative z-10">
        <Download size={26} strokeWidth={3} className="group-hover/item:scale-110 transition-transform duration-200" />
      </span>
    </button>
  );
}

