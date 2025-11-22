'use client';

import { useState, useTransition, useEffect } from 'react';
import { useFormState } from 'react-dom';
import Image from 'next/image';
import { X, CheckCircle2, Calendar, User, Briefcase, FileText } from 'lucide-react';
import { validateReportAction } from './actions';

type Report = {
  id: string;
  task_id: string | null;
  worker_id: string | null;
  photo_url: string | null;
  description: string | null;
  created_at: string;
};

type Task = {
  id: string;
  title: string;
  site_id: string;
  status: string;
};

type Worker = {
  name: string;
  role: string | null;
};

type Props = {
  report: Report;
  task: Task | null;
  worker: Worker | null;
  siteName: string;
  isOpen: boolean;
  onClose: () => void;
};

export function ReportDetailModal({
  report,
  task,
  worker,
  siteName,
  isOpen,
  onClose,
}: Props) {
  const [state, formAction] = useFormState(validateReportAction, {});
  const [isPending, startTransition] = useTransition();
  const [isValidated, setIsValidated] = useState(task?.status === 'done');

  // Mettre à jour isValidated quand la tâche change ou quand la validation réussit
  useEffect(() => {
    if (task?.status === 'done') {
      setIsValidated(true);
    }
  }, [task?.status]);

  useEffect(() => {
    if (state?.success) {
      setIsValidated(true);
      // Fermer le modal après un court délai
      const timer = setTimeout(() => {
        onClose();
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [state?.success, onClose]);


  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div 
        className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl border border-zinc-200 bg-white shadow-xl dark:border-zinc-800 dark:bg-zinc-900"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-zinc-200 bg-white px-6 py-4 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">
            Détails du rapport
          </h2>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-900 dark:hover:bg-zinc-800 dark:hover:text-white"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Tâche */}
          {task && (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <FileText className="h-5 w-5 text-zinc-400" />
                <h3 className="text-sm font-semibold text-zinc-500 dark:text-zinc-400">
                  Tâche
                </h3>
              </div>
              <p className="text-lg font-semibold text-zinc-900 dark:text-white">
                {task.title}
              </p>
              {isValidated && (
                <div className="mt-2 flex items-center gap-2 text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 className="h-4 w-4" />
                  <span className="text-sm font-medium">Tâche terminée</span>
                </div>
              )}
            </div>
          )}

          {/* Chantier */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Briefcase className="h-5 w-5 text-zinc-400" />
              <h3 className="text-sm font-semibold text-zinc-500 dark:text-zinc-400">
                Chantier
              </h3>
            </div>
            <p className="text-base text-zinc-900 dark:text-white">{siteName}</p>
          </div>

          {/* Employé */}
          {worker && (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <User className="h-5 w-5 text-zinc-400" />
                <h3 className="text-sm font-semibold text-zinc-500 dark:text-zinc-400">
                  Employé
                </h3>
              </div>
              <p className="text-base text-zinc-900 dark:text-white">
                {worker.name}
                {worker.role && (
                  <span className="ml-2 text-sm text-zinc-500 dark:text-zinc-400">
                    • {worker.role}
                  </span>
                )}
              </p>
            </div>
          )}

          {/* Date */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Calendar className="h-5 w-5 text-zinc-400" />
              <h3 className="text-sm font-semibold text-zinc-500 dark:text-zinc-400">
                Date
              </h3>
            </div>
            <p className="text-base text-zinc-900 dark:text-white">
              {new Date(report.created_at).toLocaleString('fr-FR', {
                day: 'numeric',
                month: 'long',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              })}
            </p>
          </div>

          {/* Description */}
          {report.description && (
            <div>
              <h3 className="text-sm font-semibold text-zinc-500 dark:text-zinc-400 mb-2">
                Description
              </h3>
              <p className="text-base text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap">
                {report.description}
              </p>
            </div>
          )}

          {/* Photo */}
          {report.photo_url && (
            <div>
              <h3 className="text-sm font-semibold text-zinc-500 dark:text-zinc-400 mb-3">
                Photo
              </h3>
              <div className="overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-700">
                <Image
                  src={report.photo_url}
                  alt="Photo du rapport"
                  width={800}
                  height={600}
                  className="w-full h-auto object-contain"
                  unoptimized
                />
              </div>
            </div>
          )}

          {/* Messages d'erreur/succès */}
          {state?.error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900/60 dark:bg-rose-900/20 dark:text-rose-200">
              {state.error}
            </div>
          )}

          {state?.success && (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800 dark:border-emerald-900/60 dark:bg-emerald-900/20 dark:text-emerald-200">
              ✓ {state.message}
            </div>
          )}
        </div>

        {/* Footer avec bouton de validation */}
        {task && !isValidated && (
          <div className="sticky bottom-0 border-t border-zinc-200 bg-white px-6 py-4 dark:border-zinc-800 dark:bg-zinc-900">
            <form
              action={(formData) => {
                startTransition(async () => {
                  await formAction(formData);
                });
              }}
            >
              <input type="hidden" name="reportId" value={report.id} />
              <input type="hidden" name="taskId" value={task.id} />
              <button
                type="submit"
                disabled={isPending}
                className="w-full rounded-lg bg-emerald-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-70 flex items-center justify-center gap-2"
              >
                <CheckCircle2 className="h-5 w-5" />
                {isPending ? 'Validation...' : 'Valider le rapport et terminer la tâche'}
              </button>
            </form>
          </div>
        )}

        {isValidated && (
          <div className="sticky bottom-0 border-t border-zinc-200 bg-emerald-50 px-6 py-4 dark:border-zinc-800 dark:bg-emerald-900/20">
            <div className="flex items-center justify-center gap-2 text-emerald-700 dark:text-emerald-400">
              <CheckCircle2 className="h-5 w-5" />
              <span className="font-semibold">Rapport validé et tâche terminée</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

