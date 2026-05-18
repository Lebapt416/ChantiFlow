'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Loader2 } from 'lucide-react';
import { deleteSiteAction, completeSiteFromListAction } from './actions';

type Site = {
  id: string;
  name: string;
  deadline: string | null;
  created_at: string;
  completed_at?: string | null;
};

type Stats = {
  tasks: number;
  done: number;
  workers: number;
};

type Props = {
  site: Site;
  stats: Stats;
};

export function SiteCard({ site, stats }: Props) {
  const router = useRouter();
  const [isDeleting, setIsDeleting] = useState(false);
  const [isCompleting, setIsCompleting] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [showCompleteConfirm, setShowCompleteConfirm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const progress = stats.tasks > 0 ? Math.round((stats.done / stats.tasks) * 100) : 0;
  const isCompleted = Boolean(site.completed_at);

  const handleDelete = async () => {
    setIsDeleting(true);
    setError(null);
    setSuccess(null);

    try {
      const result = await deleteSiteAction(site.id);

      if (result.success) {
        setSuccess(result.message || 'Chantier supprimé avec succès.');
        setShowDeleteConfirm(false);
        setTimeout(() => {
          router.push('/sites');
        }, 1000);
      } else {
        setError(result.error || 'Erreur lors de la suppression.');
      }
    } catch (err) {
      setError('Erreur lors de la suppression.');
      console.error(err);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleComplete = async () => {
    setIsCompleting(true);
    setError(null);
    setSuccess(null);

    try {
      const result = await completeSiteFromListAction(site.id);

      if (result.success) {
        setSuccess(result.message || 'Chantier terminé avec succès.');
        setShowCompleteConfirm(false);
        setTimeout(() => {
          router.push('/sites');
        }, 1500);
      } else {
        setError(result.error || 'Erreur lors de la terminaison.');
      }
    } catch (err) {
      setError('Erreur lors de la terminaison.');
      console.error(err);
    } finally {
      setIsCompleting(false);
    }
  };

  return (
    <div className={`flex-shrink-0 w-80 border border-rule-soft bg-paper transition-colors duration-150 ${isCompleted ? 'opacity-60' : ''}`}>
      <Link href={`/site/${site.id}/dashboard`} className="block p-5">
        {/* Header row */}
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-1">Chantier</p>
            <h3 className="font-serif text-[20px] font-normal text-ink leading-tight" style={{fontVariationSettings: '"opsz" 60'}}>{site.name}</h3>
          </div>
          {isCompleted ? (
            <span className="border border-green font-mono text-[9px] uppercase tracking-widest px-2 py-1 text-green flex-shrink-0">Terminé</span>
          ) : (
            <span className="border border-rule-soft font-mono text-[9px] uppercase tracking-widest px-2 py-1 text-ink-3 flex-shrink-0">Actif</span>
          )}
        </div>

        {/* Progress bar */}
        <div className="h-0.5 bg-paper-2 border border-rule-soft/50 mb-3">
          <div className="h-full bg-orange transition-all duration-300" style={{width: `${progress}%`}} />
        </div>

        {/* Stats row */}
        <div className="flex items-center gap-6 font-mono text-[10px] uppercase tracking-widest text-ink-3">
          <span>{stats.tasks} tâches · {stats.done} terminées</span>
          {stats.workers > 0 && <span>{stats.workers} travailleurs</span>}
          {site.deadline && <span>Livraison {new Date(site.deadline).toLocaleDateString('fr-FR')}</span>}
        </div>
      </Link>

      {/* Messages d'erreur/succès */}
      {error && (
        <div className="mx-5 mb-3 border border-danger bg-paper-2 p-2 font-mono text-[10px] text-danger">
          {error}
        </div>
      )}

      {success && (
        <div className="mx-5 mb-3 border border-rule-soft bg-paper-2 p-2 font-mono text-[10px] text-ink">
          {success}
        </div>
      )}

      {/* Boutons d'action */}
      {!isCompleted && (
        <div className="flex border-t border-rule-soft">
          <button
            onClick={() => setShowCompleteConfirm(true)}
            disabled={isCompleting || isDeleting}
            className="flex-1 flex items-center justify-center gap-2 border-r border-rule-soft px-3 py-2 font-mono text-[10px] uppercase tracking-widest text-ink-2 hover:text-ink hover:bg-paper-2 transition-colors disabled:cursor-not-allowed disabled:opacity-70"
          >
            {isCompleting ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                Terminaison...
              </>
            ) : (
              'Terminer'
            )}
          </button>

          <button
            onClick={() => setShowDeleteConfirm(true)}
            disabled={isDeleting || isCompleting}
            className="flex-1 flex items-center justify-center gap-2 px-3 py-2 font-mono text-[10px] uppercase tracking-widest border-danger text-danger hover:bg-paper-2 transition-colors disabled:cursor-not-allowed disabled:opacity-70"
          >
            {isDeleting ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                Suppression...
              </>
            ) : (
              'Supprimer'
            )}
          </button>
        </div>
      )}

      {isCompleted && (
        <div className="border-t border-rule-soft px-5 py-2 font-mono text-[10px] uppercase tracking-widest text-green text-center">
          ✓ Chantier terminé
        </div>
      )}

      {/* Modal de confirmation suppression */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-md border border-rule-soft bg-paper p-6">
            <h3 className="font-serif text-[20px] text-ink mb-2">
              Supprimer le chantier
            </h3>
            <p className="text-sm text-ink-2 mb-4">
              Êtes-vous sûr de vouloir supprimer le chantier <strong>{site.name}</strong> ?
              Cette action est irréversible et supprimera toutes les données associées (tâches, employés, rapports).
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="flex-1 border border-rule-soft px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-ink-2 hover:text-ink transition-colors"
              >
                Annuler
              </button>
              <button
                onClick={handleDelete}
                disabled={isDeleting}
                className="flex-1 border border-danger px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-danger hover:bg-paper-2 transition-colors disabled:cursor-not-allowed disabled:opacity-70"
              >
                {isDeleting ? 'Suppression...' : 'Supprimer'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Modal de confirmation terminaison */}
      {showCompleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-md border border-rule-soft bg-paper p-6">
            <h3 className="font-serif text-[20px] text-ink mb-2">
              Terminer le chantier
            </h3>
            <p className="text-sm text-ink-2 mb-4">
              Êtes-vous sûr de vouloir terminer le chantier <strong>{site.name}</strong> ?
              Les employés seront retirés du chantier et recevront un email de notification.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setShowCompleteConfirm(false)}
                className="flex-1 border border-rule-soft px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-ink-2 hover:text-ink transition-colors"
              >
                Annuler
              </button>
              <button
                onClick={handleComplete}
                disabled={isCompleting}
                className="flex-1 border border-orange px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-orange hover:bg-paper-2 transition-colors disabled:cursor-not-allowed disabled:opacity-70"
              >
                {isCompleting ? 'Terminaison...' : 'Terminer'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
