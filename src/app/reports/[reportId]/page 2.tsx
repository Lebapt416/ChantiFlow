import { redirect, notFound } from 'next/navigation';
import Image from 'next/image';
import { AppShell } from '@/components/app-shell';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { CheckCircle2, Calendar, User, Briefcase, FileText, ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import { ValidateReportButton } from './validate-report-button';

type Params = {
  params: Promise<{
    reportId: string;
  }>;
};

export default async function ReportDetailPage({ params }: Params) {
  const { reportId } = await params;

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  // Récupérer le rapport
  const { data: report } = await supabase
    .from('reports')
    .select('id, task_id, worker_id, photo_url, description, created_at')
    .eq('id', reportId)
    .single();

  if (!report) {
    notFound();
  }

  // Récupérer la tâche
  const { data: task } = report.task_id
    ? await supabase
        .from('tasks')
        .select('id, title, site_id, status')
        .eq('id', report.task_id)
        .single()
    : { data: null };

  if (!task) {
    notFound();
  }

  // Vérifier que le chantier appartient à l'utilisateur
  const { data: site } = await supabase
    .from('sites')
    .select('id, name, created_by')
    .eq('id', task.site_id)
    .single();

  if (!site || site.created_by !== user.id) {
    notFound();
  }

  // Récupérer le worker
  const { data: worker } = report.worker_id
    ? await supabase
        .from('workers')
        .select('id, name, email, role')
        .eq('id', report.worker_id)
        .single()
    : { data: null };

  const isTaskDone = task.status === 'done';

  return (
    <AppShell
      heading={`Rapport - ${task.title}`}
      subheading={site.name}
      userEmail={user.email}
      primarySite={site}
    >
      <div className="space-y-6">
        {/* Bouton retour */}
        <Link
          href="/reports"
          className="inline-flex items-center gap-2 text-sm text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 transition"
        >
          <ArrowLeft className="h-4 w-4" />
          Retour aux rapports
        </Link>

        {/* Carte principale */}
        <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          {/* En-tête avec statut */}
          <div className="mb-6 flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-2xl font-bold text-zinc-900 dark:text-white">
                  {task.title}
                </h1>
                {isTaskDone && (
                  <div className="flex items-center gap-2 rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    Tâche terminée
                  </div>
                )}
              </div>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                {site.name}
              </p>
            </div>
          </div>

          {/* Informations */}
          <div className="grid gap-4 md:grid-cols-2 mb-6">
            {/* Employé */}
            {worker && (
              <div className="flex items-start gap-3 rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50">
                <User className="h-5 w-5 text-zinc-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-xs font-semibold text-zinc-500 dark:text-zinc-400 mb-1">
                    Employé
                  </p>
                  <p className="text-sm font-medium text-zinc-900 dark:text-white">
                    {worker.name}
                  </p>
                  {worker.role && (
                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                      {worker.role}
                    </p>
                  )}
                  {worker.email && (
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      {worker.email}
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Date */}
            <div className="flex items-start gap-3 rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50">
              <Calendar className="h-5 w-5 text-zinc-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-xs font-semibold text-zinc-500 dark:text-zinc-400 mb-1">
                  Date d'envoi
                </p>
                <p className="text-sm font-medium text-zinc-900 dark:text-white">
                  {new Date(report.created_at ?? '').toLocaleString('fr-FR', {
                    day: 'numeric',
                    month: 'long',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </p>
              </div>
            </div>
          </div>

          {/* Description */}
          {report.description && (
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <FileText className="h-5 w-5 text-zinc-400" />
                <h2 className="text-sm font-semibold text-zinc-500 dark:text-zinc-400">
                  Description
                </h2>
              </div>
              <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50">
                <p className="text-sm text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap">
                  {report.description}
                </p>
              </div>
            </div>
          )}

          {/* Photo */}
          {report.photo_url && (
            <div className="mb-6">
              <h2 className="text-sm font-semibold text-zinc-500 dark:text-zinc-400 mb-3">
                Photo
              </h2>
              <div className="overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-700">
                <Image
                  src={report.photo_url}
                  alt="Photo du rapport"
                  width={1200}
                  height={800}
                  className="w-full h-auto object-contain"
                  unoptimized
                />
              </div>
            </div>
          )}

          {/* Bouton de validation */}
          {!isTaskDone && (
            <div className="pt-6 border-t border-zinc-200 dark:border-zinc-700">
              <ValidateReportButton reportId={report.id} taskId={task.id} />
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}

