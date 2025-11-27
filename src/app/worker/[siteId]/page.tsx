import { WorkerSiteShell } from '../components/worker-site-shell';

type Params = {
  params: Promise<{
    siteId: string;
  }>;
};

export default async function WorkerMobilePage({ params }: Params) {
  const { siteId } = await params;
  return (
    <div className="min-h-screen bg-zinc-50 pb-16 dark:bg-zinc-900">
      <WorkerSiteShell siteId={siteId} />
    </div>
  );
}

