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
    if (window.matchMedia('(display-mode: standalone)').matches) {
      alert('✅ L\'application est déjà installée !');
      return;
    }

    if (!deferredPrompt) {
      // Instructions pour différents navigateurs
      const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
      const isSafari = /Safari/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent);
      const isAndroid = /Android/.test(navigator.userAgent);
      const isChrome = /Chrome/.test(navigator.userAgent) && !/Edg/.test(navigator.userAgent);
      const isEdge = /Edg/.test(navigator.userAgent);
      
      let message = '📱 Pour installer l\'application :\n\n';
      
      if (isIOS || (isSafari && isIOS)) {
        message += 'Sur iOS (Safari) :\n';
        message += '1. Appuyez sur le bouton Partager (📤)\n';
        message += '2. Faites défiler et sélectionnez "Sur l\'écran d\'accueil"\n';
        message += '3. Appuyez sur "Ajouter"\n\n';
        message += 'L\'icône ChantiFlow apparaîtra sur votre écran d\'accueil !';
      } else if (isAndroid) {
        message += 'Sur Android :\n';
        message += '1. Ouvrez le menu du navigateur (⋮ en haut à droite)\n';
        message += '2. Sélectionnez "Installer l\'application" ou "Ajouter à l\'écran d\'accueil"\n';
        message += '3. Confirmez l\'installation\n\n';
        message += 'L\'application sera installée sur votre appareil !';
      } else if (isChrome) {
        message += 'Sur Chrome (PC/Mac) :\n';
        message += '1. Cherchez l\'icône d\'installation (➕) dans la barre d\'adresse\n';
        message += '2. Cliquez dessus et sélectionnez "Installer"\n';
        message += 'OU utilisez le menu (⋮) > "Installer ChantiFlow"\n\n';
        message += 'L\'application s\'ouvrira comme une application native !';
      } else if (isEdge) {
        message += 'Sur Edge (PC/Mac) :\n';
        message += '1. Cherchez l\'icône d\'installation (➕) dans la barre d\'adresse\n';
        message += '2. Cliquez dessus et sélectionnez "Installer"\n';
        message += 'OU utilisez le menu (⋯) > "Applications" > "Installer ce site en tant qu\'application"\n\n';
        message += 'L\'application s\'ouvrira comme une application native !';
      } else {
        message += 'Pour installer l\'application :\n';
        message += '1. Utilisez le menu de votre navigateur\n';
        message += '2. Cherchez "Installer" ou "Ajouter à l\'écran d\'accueil"\n';
        message += '3. Suivez les instructions de votre navigateur\n\n';
        message += 'L\'application fonctionnera hors ligne une fois installée !';
      }
      
      alert(message);
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

