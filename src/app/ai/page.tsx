import { redirect } from 'next/navigation';

export const metadata = {
  title: 'IA de Planification | ChantiFlow',
};

export default async function AIPage() {
  // Rediriger vers la page planning qui contient maintenant toutes les fonctionnalités
  redirect('/planning');
}

