'use client';

import { useState, useEffect } from 'react';
import { AppShell } from '@/components/app-shell';
import { SystemTestReport, TestResult } from '@/lib/system-test';
import { CheckCircle2, XCircle, AlertCircle, Loader2, RefreshCw } from 'lucide-react';

export default function SystemTestPage() {
  const [report, setReport] = useState<SystemTestReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runTests = async () => {
    setLoading(true);
    setError(null);
    setReport(null);

    try {
      const response = await fetch('/api/admin/system-test');
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Erreur lors de l\'exécution des tests');
      }

      const data: SystemTestReport = await response.json();
      setReport(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur inconnue');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runTests();
  }, []);

  const getScoreColor = (score: number) => {
    if (score >= 18) return 'text-emerald-600 dark:text-emerald-400';
    if (score >= 15) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getScoreBgColor = (score: number) => {
    if (score >= 18) return 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800';
    if (score >= 15) return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
    return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
  };

  const getStatusIcon = (status: TestResult['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle2 className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />;
      case 'error':
        return <XCircle className="h-5 w-5 text-red-600 dark:text-red-400" />;
      case 'warning':
        return <AlertCircle className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />;
    }
  };

  const getStatusColor = (status: TestResult['status']) => {
    switch (status) {
      case 'success':
        return 'border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/20';
      case 'error':
        return 'border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20';
      case 'warning':
        return 'border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20';
    }
  };

  return (
    <AppShell
      heading="Tests système"
      subheading="Vérification du bon fonctionnement de l'application"
    >
      <div className="space-y-6">
        {/* En-tête avec bouton de rafraîchissement */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-white">
              Rapport de tests système
            </h2>
            <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
              Dernière exécution: {report?.timestamp ? new Date(report.timestamp).toLocaleString('fr-FR') : 'Jamais'}
            </p>
          </div>
          <button
            onClick={runTests}
            disabled={loading}
            className="flex items-center gap-2 rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-800 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Exécution...</span>
              </>
            ) : (
              <>
                <RefreshCw className="h-4 w-4" />
                <span>Relancer les tests</span>
              </>
            )}
          </button>
        </div>

        {/* Erreur */}
        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-900/20">
            <div className="flex items-center gap-2">
              <XCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
              <p className="text-sm font-medium text-red-900 dark:text-red-200">
                {error}
              </p>
            </div>
          </div>
        )}

        {/* Score global */}
        {report && (
          <div className={`rounded-2xl border p-6 ${getScoreBgColor(report.overallScore)}`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-zinc-600 dark:text-zinc-400">
                  Note globale
                </p>
                <p className={`mt-2 text-4xl font-bold ${getScoreColor(report.overallScore)}`}>
                  {report.overallScore.toFixed(1)} / 20
                </p>
              </div>
              <div className="text-right">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                    <span className="text-sm text-zinc-600 dark:text-zinc-400">
                      {report.passedTests} réussi{report.passedTests > 1 ? 's' : ''}
                    </span>
                  </div>
                  {report.warningTests > 0 && (
                    <div className="flex items-center gap-2">
                      <AlertCircle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
                      <span className="text-sm text-zinc-600 dark:text-zinc-400">
                        {report.warningTests} avertissement{report.warningTests > 1 ? 's' : ''}
                      </span>
                    </div>
                  )}
                  {report.failedTests > 0 && (
                    <div className="flex items-center gap-2">
                      <XCircle className="h-4 w-4 text-red-600 dark:text-red-400" />
                      <span className="text-sm text-zinc-600 dark:text-zinc-400">
                        {report.failedTests} échec{report.failedTests > 1 ? 's' : ''}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Liste des tests */}
        {report && (
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
              Détails des tests ({report.totalTests})
            </h3>
            {report.tests.map((test, index) => (
              <div
                key={index}
                className={`rounded-xl border p-4 ${getStatusColor(test.status)}`}
              >
                <div className="flex items-start gap-3">
                  {getStatusIcon(test.status)}
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium text-zinc-900 dark:text-white">
                        {test.name}
                      </p>
                      {test.duration !== undefined && (
                        <span className="text-xs text-zinc-500 dark:text-zinc-400">
                          {test.duration}ms
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
                      {test.message}
                    </p>
                    {test.details && (
                      <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-500">
                        {test.details}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* État de chargement initial */}
        {!report && !error && loading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-zinc-400" />
          </div>
        )}
      </div>
    </AppShell>
  );
}

