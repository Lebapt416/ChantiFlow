'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { 
  TestTube, 
  CheckCircle2, 
  XCircle, 
  AlertCircle, 
  Loader2, 
  ArrowLeft,
  Clock,
  Trash2
} from 'lucide-react';

type TestResult = {
  name: string;
  status: 'success' | 'error' | 'warning';
  message: string;
  details?: string;
  duration?: number;
};

type TestReport = {
  summary: {
    score: number;
    grade: string;
    totalTests: number;
    successTests: number;
    errorTests: number;
    warningTests: number;
    totalDuration: number;
  };
  results: TestResult[];
  timestamp: string;
};

const STORAGE_KEY = 'chantiflow_system_test_history';

export default function SystemTestPage() {
  const router = useRouter();
  const [currentTest, setCurrentTest] = useState<TestReport | null>(null);
  const [testHistory, setTestHistory] = useState<TestReport[]>([]);
  const [isTesting, setIsTesting] = useState(false);
  const [selectedHistoryIndex, setSelectedHistoryIndex] = useState<number | null>(null);

  // Charger l'historique au montage
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const history = JSON.parse(stored) as TestReport[];
        setTestHistory(history);
        // Afficher le dernier test par défaut
        if (history.length > 0) {
          setCurrentTest(history[0]);
          setSelectedHistoryIndex(0);
        }
      } catch (error) {
        console.error('Erreur lors du chargement de l\'historique:', error);
      }
    }
  }, []);

  // Fonction pour sauvegarder l'historique
  const saveToHistory = (report: TestReport) => {
    const newHistory = [report, ...testHistory].slice(0, 50); // Garder max 50 tests
    setTestHistory(newHistory);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newHistory));
  };

  // Fonction pour lancer les tests
  const handleRunTests = async () => {
    setIsTesting(true);
    setCurrentTest(null);
    setSelectedHistoryIndex(null);

    try {
      const response = await fetch('/api/system/test');
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Erreur lors du test');
      }

      const report: TestReport = {
        summary: data.summary,
        results: data.results,
        timestamp: data.timestamp,
      };

      setCurrentTest(report);
      saveToHistory(report);
      setSelectedHistoryIndex(0);
    } catch (error) {
      console.error('Erreur test système:', error);
      const errorReport: TestReport = {
        summary: {
          score: 0,
          grade: 'Erreur',
          totalTests: 0,
          successTests: 0,
          errorTests: 1,
          warningTests: 0,
          totalDuration: 0,
        },
        results: [{
          name: 'Test système',
          status: 'error',
          message: error instanceof Error ? error.message : 'Erreur inconnue',
        }],
        timestamp: new Date().toISOString(),
      };
      setCurrentTest(errorReport);
      saveToHistory(errorReport);
    } finally {
      setIsTesting(false);
    }
  };

  // Fonction pour charger un test de l'historique
  const loadHistoryTest = (index: number) => {
    setCurrentTest(testHistory[index]);
    setSelectedHistoryIndex(index);
  };

  // Fonction pour supprimer l'historique
  const clearHistory = () => {
    if (confirm('Êtes-vous sûr de vouloir supprimer tout l\'historique des tests ?')) {
      localStorage.removeItem(STORAGE_KEY);
      setTestHistory([]);
      setCurrentTest(null);
      setSelectedHistoryIndex(null);
    }
  };

  const displayReport = currentTest || (selectedHistoryIndex !== null ? testHistory[selectedHistoryIndex] : null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950 text-white">
      {/* Header */}
      <div className="border-b border-zinc-800 bg-black/80 backdrop-blur">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/analytics"
                className="p-2 rounded-lg hover:bg-zinc-800 transition-colors"
                title="Retour au dashboard"
              >
                <ArrowLeft className="h-5 w-5" />
              </Link>
              <div>
                <h1 className="text-3xl font-bold flex items-center gap-3">
                  <TestTube className="h-8 w-8 text-blue-400" />
                  Tests Système
                </h1>
                <p className="mt-1 text-sm text-zinc-400">
                  Vérifiez l'état de fonctionnement de tous les composants
                </p>
              </div>
            </div>
            <button
              onClick={handleRunTests}
              disabled={isTesting}
              className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isTesting ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Test en cours...
                </>
              ) : (
                <>
                  <TestTube className="h-5 w-5" />
                  Lancer les tests
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar - Historique */}
          <div className="lg:col-span-1">
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur sticky top-8">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Clock className="h-5 w-5" />
                  Historique
                </h2>
                {testHistory.length > 0 && (
                  <button
                    onClick={clearHistory}
                    className="p-1.5 rounded-lg hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-red-400"
                    title="Supprimer l'historique"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                )}
              </div>

              {testHistory.length === 0 ? (
                <p className="text-sm text-zinc-500 text-center py-8">
                  Aucun test effectué
                </p>
              ) : (
                <div className="space-y-2 max-h-[600px] overflow-y-auto">
                  {testHistory.map((test, index) => (
                    <button
                      key={index}
                      onClick={() => loadHistoryTest(index)}
                      className={`w-full text-left p-3 rounded-lg border transition-colors ${
                        selectedHistoryIndex === index
                          ? 'border-blue-500 bg-blue-500/10'
                          : 'border-zinc-800 bg-zinc-950/50 hover:bg-zinc-800/50'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className={`text-xs font-semibold ${
                          test.summary.score >= 90
                            ? 'text-emerald-400'
                            : test.summary.score >= 75
                            ? 'text-blue-400'
                            : test.summary.score >= 50
                            ? 'text-yellow-400'
                            : 'text-red-400'
                        }`}>
                          {test.summary.score}/100
                        </span>
                        <span className="text-xs text-zinc-500">
                          {new Date(test.timestamp).toLocaleTimeString('fr-FR', {
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </span>
                      </div>
                      <p className="text-xs text-zinc-400 mb-1">
                        {new Date(test.timestamp).toLocaleDateString('fr-FR', {
                          day: '2-digit',
                          month: 'short',
                        })}
                      </p>
                      <div className="flex items-center gap-2 text-xs">
                        <span className="text-emerald-400">
                          {test.summary.successTests} ✓
                        </span>
                        {test.summary.errorTests > 0 && (
                          <span className="text-red-400">
                            {test.summary.errorTests} ✗
                          </span>
                        )}
                        {test.summary.warningTests > 0 && (
                          <span className="text-yellow-400">
                            {test.summary.warningTests} ⚠
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Main Content - Résultats */}
          <div className="lg:col-span-3">
            {!displayReport && !isTesting ? (
              <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-12 backdrop-blur text-center">
                <TestTube className="h-16 w-16 text-zinc-600 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">
                  Aucun test disponible
                </h3>
                <p className="text-zinc-400 mb-6">
                  Cliquez sur "Lancer les tests" pour commencer
                </p>
                <button
                  onClick={handleRunTests}
                  className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold transition-colors inline-flex items-center gap-2"
                >
                  <TestTube className="h-5 w-5" />
                  Lancer les tests
                </button>
              </div>
            ) : isTesting ? (
              <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-12 backdrop-blur text-center">
                <Loader2 className="h-16 w-16 text-blue-400 mx-auto mb-4 animate-spin" />
                <h3 className="text-xl font-semibold text-white mb-2">
                  Tests en cours...
                </h3>
                <p className="text-zinc-400">
                  Veuillez patienter pendant l'exécution des tests
                </p>
              </div>
            ) : displayReport ? (
              <div className="space-y-6">
                {/* Résumé global */}
                <div className={`rounded-2xl border-2 p-6 ${
                  displayReport.summary.score >= 90
                    ? 'border-emerald-500/50 bg-emerald-500/10'
                    : displayReport.summary.score >= 75
                    ? 'border-blue-500/50 bg-blue-500/10'
                    : displayReport.summary.score >= 50
                    ? 'border-yellow-500/50 bg-yellow-500/10'
                    : 'border-red-500/50 bg-red-500/10'
                }`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-xl font-bold text-white mb-2">
                        Note globale: {displayReport.summary.score}/100
                      </h3>
                      <p className={`text-3xl font-bold ${
                        displayReport.summary.score >= 90
                          ? 'text-emerald-400'
                          : displayReport.summary.score >= 75
                          ? 'text-blue-400'
                          : displayReport.summary.score >= 50
                          ? 'text-yellow-400'
                          : 'text-red-400'
                      }`}>
                        {displayReport.summary.grade}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-zinc-400 mb-1">
                        {displayReport.summary.successTests} réussis
                      </p>
                      <p className="text-sm text-zinc-400 mb-1">
                        {displayReport.summary.errorTests} erreurs
                      </p>
                      {displayReport.summary.warningTests > 0 && (
                        <p className="text-sm text-yellow-400 mb-1">
                          {displayReport.summary.warningTests} avertissements
                        </p>
                      )}
                      <p className="text-xs text-zinc-500 mt-2">
                        Durée: {displayReport.summary.totalDuration}ms
                      </p>
                      <p className="text-xs text-zinc-500 mt-1">
                        {new Date(displayReport.timestamp).toLocaleString('fr-FR')}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Détail des tests */}
                <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
                  <h3 className="text-lg font-semibold text-white mb-4">
                    Détail des tests ({displayReport.results.length})
                  </h3>
                  <div className="space-y-3">
                    {displayReport.results.map((result, index) => (
                      <div
                        key={index}
                        className={`rounded-lg border p-4 ${
                          result.status === 'success'
                            ? 'border-emerald-500/30 bg-emerald-500/5'
                            : result.status === 'error'
                            ? 'border-red-500/30 bg-red-500/5'
                            : 'border-yellow-500/30 bg-yellow-500/5'
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          {result.status === 'success' && (
                            <CheckCircle2 className="h-5 w-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                          )}
                          {result.status === 'error' && (
                            <XCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                          )}
                          {result.status === 'warning' && (
                            <AlertCircle className="h-5 w-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                          )}
                          <div className="flex-1">
                            <h4 className="font-semibold text-white mb-1">{result.name}</h4>
                            <p className={`text-sm ${
                              result.status === 'success'
                                ? 'text-emerald-300'
                                : result.status === 'error'
                                ? 'text-red-300'
                                : 'text-yellow-300'
                            }`}>
                              {result.message}
                            </p>
                            {result.details && (
                              <p className="text-xs text-zinc-400 mt-2 whitespace-pre-wrap">
                                {result.details}
                              </p>
                            )}
                            {result.duration !== undefined && (
                              <p className="text-xs text-zinc-500 mt-1">
                                Durée: {result.duration}ms
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

