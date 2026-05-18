'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { X } from 'lucide-react';
import { CreateSiteForm } from '../dashboard/create-site-form';
import Link from 'next/link';

type Props = {
  canCreate: boolean;
  limitReason?: string;
  currentCount: number;
  maxSites: number;
};

export function CreateSiteCard({ canCreate, limitReason, currentCount, maxSites }: Props) {
  const [isFormOpen, setIsFormOpen] = useState(false);
  const router = useRouter();

  function handleSuccess() {
    setIsFormOpen(false);
    router.push('/sites');
  }

  if (!canCreate) {
    return (
      <div className="flex-shrink-0 w-80 border border-warn bg-paper-2 p-6">
        <div className="flex h-full flex-col items-start justify-center">
          <p className="font-mono text-[10px] uppercase tracking-widest text-warn mb-2">Limite atteinte</p>
          <p className="text-sm text-ink-2 mb-4">
            {limitReason || `Vous avez atteint la limite de ${maxSites} chantier${maxSites > 1 ? 's' : ''}.`}
          </p>
          <Link
            href="/account"
            className="border border-rule-soft px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-ink-2 hover:text-ink hover:bg-paper-2 transition-colors"
          >
            Passer au plan supérieur →
          </Link>
        </div>
      </div>
    );
  }

  if (isFormOpen) {
    return (
      <div className="flex-shrink-0 w-80 border border-rule-soft bg-paper p-6">
        <div className="mb-4 flex items-center justify-between">
          <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3">Nouveau chantier</p>
          <button
            type="button"
            onClick={() => setIsFormOpen(false)}
            className="p-1 text-ink-3 hover:text-ink transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <CreateSiteForm onSuccess={handleSuccess} />
        <p className="mt-2 font-mono text-[10px] uppercase tracking-widest text-ink-3">
          {currentCount}/{maxSites === Infinity ? '∞' : maxSites} chantier{maxSites > 1 ? 's' : ''} utilisé{maxSites > 1 ? 's' : ''}
        </p>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={() => setIsFormOpen(true)}
      className="flex-shrink-0 w-80 block border border-dashed border-rule text-ink-2 hover:bg-paper-2 hover:text-ink transition-colors duration-150 p-5 min-h-[120px] flex items-center gap-4"
    >
      <span className="font-mono text-[28px] text-ink-3">+</span>
      <div className="text-left">
        <p className="font-sans font-medium text-[15px] text-ink">Nouveau chantier</p>
        <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mt-1">Créer →</p>
        <p className="font-mono text-[10px] text-ink-3 mt-1">
          {currentCount}/{maxSites === Infinity ? '∞' : maxSites} utilisé{maxSites > 1 ? 's' : ''}
        </p>
      </div>
    </button>
  );
}
