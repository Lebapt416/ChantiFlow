'use client';

import { X } from 'lucide-react';
import { useEffect, useState } from 'react';

interface InstallPwaModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function InstallPwaModal({ isOpen, onClose }: InstallPwaModalProps) {
  const [instructions, setInstructions] = useState('');

  useEffect(() => {
    if (!isOpen) return;

    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
    const isSafari = /Safari/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent);
    const isAndroid = /Android/.test(navigator.userAgent);
    const isChrome = /Chrome/.test(navigator.userAgent) && !/Edg/.test(navigator.userAgent);
    const isEdge = /Edg/.test(navigator.userAgent);
    const isMac = /Mac/.test(navigator.userAgent);

    let steps: string[] = [];
    let title = '📱 Installer ChantiFlow';

    if (isIOS || (isSafari && isIOS)) {
      title = '📱 Installer sur iOS';
      steps = [
        'Appuyez sur le bouton Partager (📤) en bas de l\'écran',
        'Faites défiler vers le bas dans le menu',
        'Sélectionnez "Sur l\'écran d\'accueil"',
        'Appuyez sur "Ajouter" en haut à droite',
      ];
    } else if (isSafari && isMac) {
      title = '📱 Installer sur Safari (Mac)';
      steps = [
        'Cliquez sur "Fichier" dans la barre de menu',
        'Sélectionnez "Ajouter à l\'écran d\'accueil..."',
        'Confirmez le nom de l\'application',
        'Cliquez sur "Ajouter"',
      ];
    } else if (isAndroid) {
      title = '📱 Installer sur Android';
      steps = [
        'Ouvrez le menu du navigateur (⋮) en haut à droite',
        'Sélectionnez "Installer l\'application" ou "Ajouter à l\'écran d\'accueil"',
        'Confirmez l\'installation',
        'L\'application apparaîtra sur votre écran d\'accueil',
      ];
    } else if (isChrome) {
      title = '📱 Installer sur Chrome';
      steps = [
        'Cherchez l\'icône d\'installation (➕) dans la barre d\'adresse',
        'Cliquez dessus et sélectionnez "Installer"',
        'OU utilisez le menu (⋮) > "Installer ChantiFlow"',
        'L\'application s\'ouvrira comme une application native',
      ];
    } else if (isEdge) {
      title = '📱 Installer sur Edge';
      steps = [
        'Cherchez l\'icône d\'installation (➕) dans la barre d\'adresse',
        'Cliquez dessus et sélectionnez "Installer"',
        'OU utilisez le menu (⋯) > "Applications" > "Installer ce site"',
        'L\'application s\'ouvrira comme une application native',
      ];
    } else {
      title = '📱 Installer l\'application';
      steps = [
        'Utilisez le menu de votre navigateur',
        'Cherchez "Installer" ou "Ajouter à l\'écran d\'accueil"',
        'Suivez les instructions de votre navigateur',
        'L\'application fonctionnera hors ligne une fois installée',
      ];
    }

    setInstructions(steps.join('\n'));
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="relative w-full max-w-md mx-4 bg-white dark:bg-zinc-900 rounded-xl shadow-2xl p-6">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
        >
          <X size={24} />
        </button>

        <h2 className="text-2xl font-bold mb-4 text-zinc-900 dark:text-white">
          Installer ChantiFlow
        </h2>

        <div className="space-y-3 mb-6">
          {instructions.split('\n').map((step, index) => (
            <div key={index} className="flex items-start gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-black dark:bg-white text-white dark:text-black flex items-center justify-center text-sm font-bold">
                {index + 1}
              </span>
              <p className="text-zinc-700 dark:text-zinc-300 leading-relaxed">
                {step}
              </p>
            </div>
          ))}
        </div>

        <div className="bg-zinc-100 dark:bg-zinc-800 rounded-lg p-4 mb-6">
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            💡 <strong>Astuce :</strong> Une fois installée, l'application fonctionnera hors ligne et s'ouvrira comme une application native !
          </p>
        </div>

        <button
          onClick={onClose}
          className="w-full rounded-md bg-black dark:bg-white text-white dark:text-black py-3 font-medium transition hover:bg-zinc-800 dark:hover:bg-zinc-100"
        >
          J'ai compris
        </button>
      </div>
    </div>
  );
}

