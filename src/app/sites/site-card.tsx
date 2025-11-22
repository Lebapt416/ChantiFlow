'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { FolderKanban, CheckCircle2, Trash2, Loader2 } from 'lucide-react';
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
        // Recharger la page après un court délai
        setTimeout(() => {
          router.refresh();
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
        // Recharger la page après un court délai
        setTimeout(() => {
          router.refresh();
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
    <div className="flex-shrink-0 w-80 rounded-2xl border border-zinc-200 bg-white p-6 shadow-lg shadow-black/5 transition hover:shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-4 flex items-start justify-between">
        <div className="flex-1">
          <div className="mb-2 flex items-center gap-2">
            <FolderKanban className="h-5 w-5 text-zinc-500 dark:text-zinc-400" />
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
              {site.name}
            </h3>
            {isCompleted && (
              <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
            )}
          </div>
          {site.deadline ? (
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Deadline : {new Date(site.deadline).toLocaleDateString('fr-FR')}
            </p>
          ) : (
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Deadline non définie
            </p>
          )}
        </div>
      </div>

      <div className="mb-4 grid grid-cols-3 gap-3">
        <div className="rounded-lg border border-zinc-100 bg-zinc-50 p-2 text-center dark:border-zinc-800 dark:bg-zinc-800">
          <p className="text-xs text-zinc-500 dark:text-zinc-400">Tâches</p>
          <p className="mt-1 text-lg font-semibold text-zinc-900 dark:text-white">
            {stats.tasks}
          </p>
        </div>
        <div className="rounded-lg border border-zinc-100 bg-zinc-50 p-2 text-center dark:border-zinc-800 dark:bg-zinc-800">
          <p className="text-xs text-zinc-500 dark:text-zinc-400">Terminées</p>
          <p className="mt-1 text-lg font-semibold text-emerald-600 dark:text-emerald-400">
            {stats.done}
          </p>
        </div>
        <div className="rounded-lg border border-zinc-100 bg-zinc-50 p-2 text-center dark:border-zinc-800 dark:bg-zinc-800">
          <p className="text-xs text-zinc-500 dark:text-zinc-400">Équipe</p>
          <p className="mt-1 text-lg font-semibold text-zinc-900 dark:text-white">
            {stats.workers}
          </p>
        </div>
      </div>

      <div className="mb-4">
        <div className="mb-1 flex items-center justify-between text-xs">
          <span className="text-zinc-500 dark:text-zinc-400">Progression</span>
          <span className="font-semibold text-zinc-900 dark:text-white">{progress}%</span>
        </div>
        <div className="h-2 rounded-full bg-zinc-200 dark:bg-zinc-800">
          <div
            className="h-full rounded-full bg-emerald-500 transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Messages d'erreur/succès */}
      {error && (
        <div className="mb-3 rounded-lg border border-rose-200 bg-rose-50 p-2 text-xs text-rose-800 dark:border-rose-900/60 dark:bg-rose-900/20 dark:text-rose-200">
          {error}
        </div>
      )}

      {success && (
        <div className="mb-3 rounded-lg border border-emerald-200 bg-emerald-50 p-2 text-xs text-emerald-800 dark:border-emerald-900/60 dark:bg-emerald-900/20 dark:text-emerald-200">
          {success}
        </div>
      )}

      {/* Boutons d'action */}
      <div className="space-y-2">
        <Link
          href={`/site/${site.id}`}
          className="block w-full rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-center text-xs font-semibold text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
        >
          Ouvrir le chantier →
        </Link>

        {!isCompleted && (
          <>
            <button
              onClick={() => setShowCompleteConfirm(true)}
              disabled={isCompleting || isDeleting}
              className="w-full flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-xs font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {isCompleting ? (
                <>
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Terminaison...
                </>
              ) : (
                <>
                  <CheckCircle2 className="h-3 w-3" />
                  Terminer
                </>
              )}
            </button>

            <button
              onClick={() => setShowDeleteConfirm(true)}
              disabled={isDeleting || isCompleting}
              className="w-full flex items-center justify-center gap-2 rounded-lg border border-rose-300 bg-white px-3 py-2 text-xs font-semibold text-rose-700 transition hover:bg-rose-50 dark:border-rose-700 dark:bg-zinc-800 dark:text-rose-400 dark:hover:bg-rose-900/20 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {isDeleting ? (
                <>
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Suppression...
                </>
              ) : (
                <>
                  <Trash2 className="h-3 w-3" />
                  Supprimer
                </>
              )}
            </button>
          </>
        )}

        {isCompleted && (
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-center text-xs font-semibold text-emerald-700 dark:border-emerald-800 dark:bg-emerald-900/20 dark:text-emerald-400">
            ✓ Chantier terminé
          </div>
        )}
      </div>

      {/* Modal de confirmation suppression */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-md rounded-2xl border border-zinc-200 bg-white p-6 shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-white mb-2">
              Supprimer le chantier
            </h3>
            <p className="text-sm text-zinc-600 dark:text-zinc-400 mb-4">
              Êtes-vous sûr de vouloir supprimer le chantier <strong>{site.name}</strong> ? 
              Cette action est irréversible et supprimera toutes les données associées (tâches, employés, rapports).
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="flex-1 rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-semibold text-zinc-700 transition hover:bg-zinc-50 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-300"
              >
                Annuler
              </button>
              <button
                onClick={handleDelete}
                disabled={isDeleting}
                className="flex-1 rounded-lg bg-rose-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-700 disabled:cursor-not-allowed disabled:opacity-70"
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
          <div className="w-full max-w-md rounded-2xl border border-zinc-200 bg-white p-6 shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-white mb-2">
              Terminer le chantier
            </h3>
            <p className="text-sm text-zinc-600 dark:text-zinc-400 mb-4">
              Êtes-vous sûr de vouloir terminer le chantier <strong>{site.name}</strong> ? 
              Les employés seront retirés du chantier et recevront un email de notification.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setShowCompleteConfirm(false)}
                className="flex-1 rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-semibold text-zinc-700 transition hover:bg-zinc-50 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-300"
              >
                Annuler
              </button>
              <button
                onClick={handleComplete}
                disabled={isCompleting}
                className="flex-1 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-70"
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

