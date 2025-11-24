import { Brain, QrCode, FileText, Camera, TrendingUp, AlertTriangle } from "lucide-react";

const featureCards = [
  { icon: Brain, title: "Planning automatique" },
  { icon: QrCode, title: "QR codes pour les équipes" },
  { icon: FileText, title: "Rapports auto" },
  { icon: Camera, title: "Photos & documents" },
  { icon: TrendingUp, title: "Suivi en temps réel" },
  { icon: AlertTriangle, title: "Alerte retards" },
];

export function FeaturesSection() {
  return (
    <section id="features" className="mx-auto max-w-7xl px-6 py-20">
      <div className="mx-auto max-w-2xl text-center">
        <h2 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
          Tout ce dont vous avez besoin
        </h2>
        <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
          Des outils puissants pour optimiser vos projets de construction
        </p>
      </div>

      <div className="mx-auto mt-12 max-w-5xl">
        <div className="rounded-2xl border border-zinc-200 bg-gradient-to-br from-zinc-50 to-zinc-100 p-4 shadow-xl dark:border-zinc-800 dark:from-zinc-900 dark:to-zinc-950">
          <div className="rounded-lg overflow-hidden bg-zinc-900">
            <div className="aspect-video bg-zinc-900 flex items-center justify-center relative">
              <div className="w-full h-full p-6 space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-zinc-800 rounded-lg p-3">
                    <div className="text-xs text-zinc-400 mb-1">CHANTIERS</div>
                    <div className="text-2xl font-bold text-white">2</div>
                    <div className="text-xs text-zinc-500">sites suivis</div>
                  </div>
                  <div className="bg-zinc-800 rounded-lg p-3">
                    <div className="text-xs text-zinc-400 mb-1">TÂCHES</div>
                    <div className="text-2xl font-bold text-white">4</div>
                    <div className="text-xs text-zinc-500">en attente</div>
                  </div>
                  <div className="bg-zinc-800 rounded-lg p-3">
                    <div className="text-xs text-zinc-400 mb-1">PROGRESSION</div>
                    <div className="text-2xl font-bold text-emerald-400">0%</div>
                    <div className="w-full bg-zinc-700 rounded-full h-1.5 mt-2">
                      <div className="bg-emerald-400 h-1.5 rounded-full" style={{ width: "0%" }} />
                    </div>
                  </div>
                </div>
                <div className="bg-zinc-800 rounded-lg p-4">
                  <div className="text-sm font-semibold text-white mb-2">Chantiers récents</div>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-zinc-300">Test chantier 2</span>
                      <span className="text-xs text-zinc-500">27/11/2025</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-zinc-300">Test chantier 1</span>
                      <span className="text-xs text-zinc-500">23/11/2025</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 text-center mt-3">
            Interface de gestion de chantier ChantiFlow
          </p>
        </div>
      </div>

      <div className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-6 sm:grid-cols-2 lg:max-w-none lg:grid-cols-3">
        {featureCards.map((feature) => (
          <div
            key={feature.title}
            className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
          >
            <div className="flex items-center gap-3">
              <feature.icon className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
              <h3 className="text-base font-semibold text-zinc-900 dark:text-white">{feature.title}</h3>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}


