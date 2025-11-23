import { redirect } from 'next/navigation';

type Params = {
  params: Promise<{
    siteId: string;
  }>;
};

/**
 * Redirige vers la page de connexion worker unifiée
 * Cette page est conservée pour la compatibilité avec les anciens liens
 */
export default async function VerifyCodePage({ params }: Params) {
  const { siteId } = await params;
  // Rediriger vers la page de connexion worker unifiée
  redirect(`/worker/login?siteId=${siteId}`);
}

