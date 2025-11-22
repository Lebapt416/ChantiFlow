import { redirect } from 'next/navigation';

export const dynamic = 'force-dynamic';

export default function Home() {
  // Toujours rediriger vers la landing page sur le site web
  // La connexion automatique sera gérée côté client pour la PWA uniquement
  redirect('/landing');
}
