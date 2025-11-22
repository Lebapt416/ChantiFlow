import { notFound, redirect } from 'next/navigation';

type Params = {
  params: Promise<{
    id: string;
  }>;
};

function isValidUuid(value: string | undefined) {
  return Boolean(value && /^[0-9a-fA-F-]{36}$/.test(value));
}

export default async function SitePage({ params }: Params) {
  const { id } = await params;

  if (!isValidUuid(id)) {
    notFound();
  }

  // Rediriger vers le dashboard du chantier
  redirect(`/site/${id}/dashboard`);
}
