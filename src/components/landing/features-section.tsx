'use client';

import { QrCode, FileText, Calendar } from "lucide-react";
import { motion } from "framer-motion";

export function FeaturesSection() {
  return (
    <section id="features" className="mx-auto max-w-7xl px-6 py-20">
      <div className="mx-auto max-w-2xl text-center mb-16">
        <h2 className="text-3xl font-extrabold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
          Tout ce dont vous avez besoin
        </h2>
        <p className="mt-4 text-lg text-zinc-700 dark:text-zinc-300 font-medium">
          Des outils puissants pour optimiser vos projets de construction
        </p>
      </div>

      {/* Bento Grid */}
      <div className="mx-auto max-w-6xl grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Carte 1 - Planning Intelligent (Grande, span-2) */}
        <motion.div
          className="md:col-span-2 rounded-3xl border border-white/20 dark:border-zinc-800/50 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md p-8 shadow-lg"
          whileHover={{
            scale: 1.05,
            borderColor: "rgba(16, 185, 129, 0.5)",
            boxShadow: "0 20px 25px -5px rgba(16, 185, 129, 0.1), 0 10px 10px -5px rgba(16, 185, 129, 0.04)",
          }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <motion.div
              className="rounded-xl bg-emerald-100 dark:bg-emerald-900/30 p-3"
              whileHover={{ rotate: [0, -10, 10, -10, 0] }}
              transition={{ duration: 0.5 }}
            >
              <Calendar className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
            </motion.div>
            <h3 className="text-2xl font-extrabold text-zinc-900 dark:text-white">
              Planning Intelligent
            </h3>
          </div>
          <p className="text-zinc-700 dark:text-zinc-300 mb-6 text-lg font-medium">
            Glissez-déposez vos tâches. Si un retard survient, l&apos;IA recalcule automatiquement la fin du chantier.
          </p>
          
          {/* Visualisation abstraite de planning */}
          <div className="space-y-3">
            {/* Ligne de temps */}
            <div className="relative h-2 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-full" style={{ width: '65%' }} />
            </div>
            
            {/* Tâches */}
            <div className="grid grid-cols-4 gap-2">
              <div className="bg-emerald-100 dark:bg-emerald-900/30 rounded-lg p-3 border-l-4 border-emerald-500">
                <div className="text-xs font-semibold text-emerald-700 dark:text-emerald-400">Maçonnerie</div>
                <div className="text-xs text-zinc-600 dark:text-zinc-400 mt-1">✓ Terminé</div>
              </div>
              <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-3 border-l-4 border-blue-500">
                <div className="text-xs font-semibold text-blue-700 dark:text-blue-400">Électricité</div>
                <div className="text-xs text-zinc-600 dark:text-zinc-400 mt-1">En cours</div>
              </div>
              <div className="bg-amber-100 dark:bg-amber-900/30 rounded-lg p-3 border-l-4 border-amber-500">
                <div className="text-xs font-semibold text-amber-700 dark:text-amber-400">Plomberie</div>
                <div className="text-xs text-zinc-600 dark:text-zinc-400 mt-1">⚠ Retard</div>
              </div>
              <div className="bg-zinc-100 dark:bg-zinc-800 rounded-lg p-3 border-l-4 border-zinc-400">
                <div className="text-xs font-semibold text-zinc-700 dark:text-zinc-400">Peinture</div>
                <div className="text-xs text-zinc-600 dark:text-zinc-400 mt-1">À venir</div>
              </div>
            </div>
            
            {/* Indicateur de recalcul IA */}
            <div className="flex items-center gap-2 text-xs text-emerald-600 dark:text-emerald-400 mt-4">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              <span className="font-medium">IA recalcule automatiquement la fin du chantier</span>
            </div>
          </div>
        </motion.div>

        {/* Carte 2 - Accès Chantier (Carrée) */}
        <motion.div
          className="rounded-3xl border border-white/20 dark:border-zinc-800/50 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md p-8 shadow-lg flex flex-col"
          whileHover={{
            scale: 1.05,
            borderColor: "rgba(59, 130, 246, 0.5)",
            boxShadow: "0 20px 25px -5px rgba(59, 130, 246, 0.1), 0 10px 10px -5px rgba(59, 130, 246, 0.04)",
          }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <motion.div
              className="rounded-xl bg-blue-100 dark:bg-blue-900/30 p-3"
              whileHover={{ rotate: [0, -10, 10, -10, 0] }}
              transition={{ duration: 0.5 }}
            >
              <QrCode className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </motion.div>
            <h3 className="text-2xl font-extrabold text-zinc-900 dark:text-white">
              Accès Chantier
            </h3>
          </div>
          <p className="text-zinc-700 dark:text-zinc-300 mb-6 text-lg flex-1 font-medium">
            QR Code unique pour chaque ouvrier.
          </p>
          
          {/* Visualisation QR Code */}
          <div className="flex items-center justify-center mt-auto">
            <div className="w-32 h-32 bg-zinc-900 dark:bg-zinc-100 rounded-lg p-4 flex items-center justify-center">
              <div className="grid grid-cols-4 gap-1 w-full h-full">
                {Array.from({ length: 16 }).map((_, i) => (
                  <div
                    key={i}
                    className={`rounded ${
                      i % 3 === 0 || i % 5 === 0
                        ? 'bg-white dark:bg-zinc-900'
                        : 'bg-zinc-900 dark:bg-white'
                    }`}
                  />
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Carte 3 - Rapports PDF (Carrée, fond noir) */}
        <motion.div
          className="rounded-3xl border border-white/10 dark:border-zinc-800/50 bg-zinc-900/90 dark:bg-black/90 backdrop-blur-md p-8 shadow-lg"
          whileHover={{
            scale: 1.05,
            borderColor: "rgba(255, 255, 255, 0.3)",
            boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2)",
          }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <motion.div
              className="rounded-xl bg-white/10 p-3"
              whileHover={{ rotate: [0, -10, 10, -10, 0] }}
              transition={{ duration: 0.5 }}
            >
              <FileText className="h-6 w-6 text-white" />
            </motion.div>
            <h3 className="text-2xl font-extrabold text-white">
              Rapports PDF
            </h3>
          </div>
          <p className="text-zinc-300 mb-6 text-lg font-medium">
            Générez vos comptes-rendus en 1 clic.
          </p>
          
          {/* Visualisation PDF */}
          <div className="mt-auto">
            <div className="bg-white/5 rounded-lg p-4 border border-white/10">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 bg-red-500 rounded flex items-center justify-center">
                  <FileText className="h-4 w-4 text-white" />
                </div>
                <div className="flex-1">
                  <div className="h-2 bg-white/20 rounded w-3/4 mb-1" />
                  <div className="h-2 bg-white/10 rounded w-1/2" />
                </div>
              </div>
              <div className="space-y-2">
                <div className="h-1 bg-white/10 rounded" />
                <div className="h-1 bg-white/10 rounded w-5/6" />
                <div className="h-1 bg-white/10 rounded w-4/6" />
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}


