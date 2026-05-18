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
          className="md:col-span-2 rounded border border-rule-soft bg-paper p-8"
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <motion.div
              className="rounded bg-paper-2 p-3"
              whileHover={{ rotate: [0, -10, 10, -10, 0] }}
              transition={{ duration: 0.5 }}
            >
              <Calendar className="h-6 w-6 text-orange" />
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
            <div className="relative h-2 bg-paper-2 rounded-full overflow-hidden">
              <div className="absolute inset-0 bg-orange rounded-full" style={{ width: '65%' }} />
            </div>
            
            {/* Tâches */}
            <div className="grid grid-cols-4 gap-2">
              <div className="bg-paper-2 rounded p-3 border-l-4 border-green">
                <div className="text-xs font-semibold text-green">Maçonnerie</div>
                <div className="text-xs text-zinc-600 dark:text-zinc-400 mt-1">✓ Terminé</div>
              </div>
              <div className="bg-paper-2 rounded p-3 border-l-4 border-blue">
                <div className="text-xs font-semibold text-blue">Électricité</div>
                <div className="text-xs text-ink-3 mt-1">En cours</div>
              </div>
              <div className="bg-paper-2 rounded p-3 border-l-4 border-warn">
                <div className="text-xs font-semibold text-warn">Plomberie</div>
                <div className="text-xs text-ink-3 mt-1">⚠ Retard</div>
              </div>
              <div className="bg-paper-2 rounded p-3 border-l-4 border-rule-soft">
                <div className="text-xs font-semibold text-ink-2">Peinture</div>
                <div className="text-xs text-ink-3 mt-1">À venir</div>
              </div>
            </div>
            
            {/* Indicateur de recalcul IA */}
            <div className="flex items-center gap-2 text-xs text-orange mt-4">
              <div className="w-2 h-2 bg-orange rounded-full animate-pulse" />
              <span className="font-medium">IA recalcule automatiquement la fin du chantier</span>
            </div>
          </div>
        </motion.div>

        {/* Carte 2 - Accès Chantier (Carrée) */}
        <motion.div
          className="rounded border border-rule-soft bg-paper p-8 flex flex-col"
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <motion.div
              className="rounded bg-paper-2 p-3"
              whileHover={{ rotate: [0, -10, 10, -10, 0] }}
              transition={{ duration: 0.5 }}
            >
              <QrCode className="h-6 w-6 text-blue" />
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
          className="rounded border border-rule bg-ink p-8"
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <motion.div
              className="rounded bg-paper/10 p-3"
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
            <div className="bg-paper/5 rounded p-4 border border-paper/10">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 bg-red-500 rounded flex items-center justify-center">
                  <FileText className="h-4 w-4 text-white" />
                </div>
                <div className="flex-1">
                  <div className="h-2 bg-paper/20 rounded w-3/4 mb-1" />
                  <div className="h-2 bg-paper/10 rounded w-1/2" />
                </div>
              </div>
              <div className="space-y-2">
                <div className="h-1 bg-paper/10 rounded" />
                <div className="h-1 bg-paper/10 rounded w-5/6" />
                <div className="h-1 bg-paper/10 rounded w-4/6" />
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}


