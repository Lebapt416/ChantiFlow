'use client';

import { useState, useEffect } from 'react';
import { X, FileText, Clock, CheckCircle2, AlertCircle, Sparkles } from 'lucide-react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';

type Task = {
  id: string;
  title: string;
  status: string;
  required_role: string | null;
  duration_hours: number | null;
  created_at: string;
  assigned_worker_id: string | null;
};

type TaskDetail = {
  task: Task;
  reportCount: number;
  aiDescription: string | null;
  isLoading: boolean;
};

type Props = {
  taskId: string;
  siteId: string;
  isOpen: boolean;
  onClose: () => void;
};

export function TaskDetailModal({ taskId, siteId, isOpen, onClose }: Props) {
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!isOpen || !taskId) return;

    async function loadTaskDetail() {
      setIsLoading(true);
      const supabase = createSupabaseBrowserClient();

      try {
        // Charger la tâche
        const { data: task } = await supabase
          .from('tasks')
          .select('id, title, status, required_role, duration_hours, created_at, assigned_worker_id')
          .eq('id', taskId)
          .single();

        if (!task) {
          setDetail(null);
          setIsLoading(false);
          return;
        }

        // Compter les rapports pour cette tâche
        const { count: reportCount } = await supabase
          .from('reports')
          .select('*', { count: 'exact', head: true })
          .eq('task_id', taskId);

        // Générer une description IA
        let aiDescription: string | null = null;
        try {
          const descriptionResponse = await fetch('/api/ai/task-description', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              taskTitle: task.title,
              requiredRole: task.required_role,
              durationHours: task.duration_hours,
              reportCount: reportCount || 0,
            }),
          });

          if (descriptionResponse.ok) {
            const data = await descriptionResponse.json();
            aiDescription = data.description;
          }
        } catch (error) {
          console.error('Erreur génération description IA:', error);
        }

        setDetail({
          task,
          reportCount: reportCount || 0,
          aiDescription,
          isLoading: false,
        });
      } catch (error) {
        console.error('Erreur chargement détails tâche:', error);
        setDetail(null);
      } finally {
        setIsLoading(false);
      }
    }

    loadTaskDetail();
  }, [isOpen, taskId, siteId]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl border border-zinc-200 bg-white shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
            Détails de la tâche
          </h2>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 sm:p-6 space-y-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-zinc-300 border-t-emerald-600"></div>
            </div>
          ) : detail ? (
            <>
              {/* Titre et statut */}
              <div>
                <h3 className="text-xl font-bold text-zinc-900 dark:text-white mb-2">
                  {detail.task.title}
                </h3>
                <div className="flex items-center gap-2">
                  {detail.task.status === 'done' ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2.5 py-1 text-xs font-semibold text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
                      <CheckCircle2 className="h-3 w-3" />
                      Terminée
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2.5 py-1 text-xs font-semibold text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
                      <Clock className="h-3 w-3" />
                      En cours
                    </span>
                  )}
                  {detail.task.required_role && (
                    <span className="rounded-full bg-zinc-100 px-2.5 py-1 text-xs font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                      {detail.task.required_role}
                    </span>
                  )}
                </div>
              </div>

              {/* Description IA */}
              {detail.aiDescription && (
                <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 dark:border-emerald-900/60 dark:bg-emerald-900/20">
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                    <h4 className="text-sm font-semibold text-emerald-900 dark:text-emerald-200">
                      Description générée par l'IA
                    </h4>
                  </div>
                  <p className="text-sm text-emerald-800 dark:text-emerald-300 leading-relaxed">
                    {detail.aiDescription}
                  </p>
                </div>
              )}

              {/* Statistiques */}
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900">
                  <div className="flex items-center gap-2 mb-1">
                    <FileText className="h-4 w-4 text-zinc-500 dark:text-zinc-400" />
                    <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
                      Rapports envoyés
                    </span>
                  </div>
                  <p className="text-2xl font-bold text-zinc-900 dark:text-white">
                    {detail.reportCount}
                  </p>
                </div>

                {detail.task.duration_hours && (
                  <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900">
                    <div className="flex items-center gap-2 mb-1">
                      <Clock className="h-4 w-4 text-zinc-500 dark:text-zinc-400" />
                      <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
                        Durée estimée
                      </span>
                    </div>
                    <p className="text-2xl font-bold text-zinc-900 dark:text-white">
                      {detail.task.duration_hours}h
                    </p>
                  </div>
                )}

                <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900">
                  <div className="flex items-center gap-2 mb-1">
                    <AlertCircle className="h-4 w-4 text-zinc-500 dark:text-zinc-400" />
                    <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
                      Date de création
                    </span>
                  </div>
                  <p className="text-sm font-semibold text-zinc-900 dark:text-white">
                    {detail.task.created_at
                      ? new Date(detail.task.created_at).toLocaleDateString('fr-FR', {
                          day: 'numeric',
                          month: 'long',
                          year: 'numeric',
                        })
                      : '---'}
                  </p>
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-12">
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Impossible de charger les détails de la tâche.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

