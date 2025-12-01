'use client';

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { motion } from "framer-motion";

function ProductScreenshot() {
  const [imageError, setImageError] = useState(false);

  if (imageError) {
    return (
      <div className="flex items-center justify-center h-full p-8">
        <div className="text-center">
          <div className="w-24 h-24 mx-auto mb-4 rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center">
            <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
            </svg>
          </div>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">Capture d&apos;écran du planning</p>
        </div>
      </div>
    );
  }

  return (
    <Image
      src="/capture-planning.png"
      alt="Capture d'écran du planning ChantiFlow"
      fill
      className="object-contain"
      priority
      onError={() => setImageError(true)}
    />
  );
}

// Animation variants pour la cascade
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
      delayChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: [0.22, 1, 0.36, 1] as [number, number, number, number],
    },
  },
};

export function HeroSection() {
  return (
    <>
      <header className="sticky top-0 z-50 border-b border-zinc-200/50 bg-white/80 backdrop-blur-md dark:border-zinc-800/50 dark:bg-zinc-950/80">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Image
                src="/logo.svg"
                alt="ChantiFlow"
                width={40}
                height={40}
                priority
                className="h-10 w-10"
              />
              <span className="text-xl font-bold text-zinc-900 dark:text-white">
                ChantiFlow
              </span>
            </div>
            <nav className="flex items-center gap-6">
              <Link
                href="/login"
                className="text-sm font-medium text-zinc-600 transition hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
              >
                Connexion
              </Link>
              <Link
                href="/login"
                className="rounded-full bg-black px-5 py-2 text-sm font-semibold text-white transition hover:bg-zinc-800 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
              >
                Commencer
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <section className="relative mx-auto max-w-7xl px-6 py-20 overflow-hidden">
        {/* Fond animé avec cercles colorés */}
        <div className="absolute inset-0 -z-10 overflow-hidden">
          <motion.div
            className="absolute left-1/4 top-1/4 w-[600px] h-[600px] bg-emerald-500/20 dark:bg-emerald-400/20 rounded-full blur-3xl"
            animate={{
              x: [0, 100, 0],
              y: [0, 50, 0],
              scale: [1, 1.2, 1],
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
          <motion.div
            className="absolute right-1/4 top-1/3 w-[500px] h-[500px] bg-blue-500/20 dark:bg-blue-400/20 rounded-full blur-3xl"
            animate={{
              x: [0, -80, 0],
              y: [0, 80, 0],
              scale: [1, 1.1, 1],
            }}
            transition={{
              duration: 25,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
          <motion.div
            className="absolute left-1/2 bottom-1/4 w-[550px] h-[550px] bg-purple-500/20 dark:bg-purple-400/20 rounded-full blur-3xl"
            animate={{
              x: [0, 60, 0],
              y: [0, -60, 0],
              scale: [1, 1.15, 1],
            }}
            transition={{
              duration: 22,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        </div>

        <motion.div
          className="mx-auto max-w-4xl text-center"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Badge */}
          <motion.div variants={itemVariants}>
            <span className="inline-block rounded-full bg-emerald-100 dark:bg-emerald-900/30 px-4 py-1.5 text-sm font-semibold text-emerald-700 dark:text-emerald-400 border border-emerald-200/50 dark:border-emerald-800/50">
              Logiciel de gestion BTP
            </span>
          </motion.div>

          {/* Titre avec gradient agressif */}
          <motion.h1
            variants={itemVariants}
            className="relative mt-6 text-5xl font-extrabold tracking-tight sm:text-6xl lg:text-7xl"
          >
            <span className="bg-gradient-to-r from-emerald-600 via-emerald-500 to-cyan-500 bg-clip-text text-transparent dark:from-emerald-400 dark:via-emerald-300 dark:to-cyan-400">
              Fini les retards de chantier.
            </span>
          </motion.h1>

          {/* Sous-titre */}
          <motion.p
            variants={itemVariants}
            className="mt-6 text-xl leading-8 text-zinc-700 dark:text-zinc-300 max-w-2xl mx-auto font-medium"
          >
            Le seul planning qui se met à jour tout seul quand une tâche prend du retard. Plus d&apos;appels le soir, plus de papier.
          </motion.p>
          
          {/* Boutons */}
          <motion.div
            variants={itemVariants}
            className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link
              href="/login"
              className="rounded-full bg-emerald-600 px-8 py-4 text-lg font-bold text-white shadow-xl shadow-emerald-500/20 transition hover:bg-emerald-700 dark:bg-emerald-500 dark:hover:bg-emerald-600 hover:scale-105"
            >
              Lancer mon 1er chantier
            </Link>
            <Link
              href="/login"
              className="rounded-full border-2 border-zinc-300/50 dark:border-zinc-700/50 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-sm px-8 py-4 text-lg font-semibold text-zinc-900 dark:text-white transition hover:bg-zinc-100/50 dark:hover:bg-zinc-800/50 hover:scale-105"
            >
              Parler à un humain
            </Link>
          </motion.div>

          {/* Product screenshot avec rotation 3D */}
          <motion.div
            variants={itemVariants}
            className="mt-16 relative"
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          >
            <motion.div
              className="relative mx-auto max-w-5xl rounded-2xl border border-white/20 dark:border-zinc-800/50 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md shadow-2xl shadow-emerald-500/20 overflow-hidden"
              style={{
                perspective: "1000px",
                transformStyle: "preserve-3d",
              }}
              whileHover={{
                rotateY: 2,
                rotateX: -2,
              }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
            >
              <div className="aspect-video relative bg-zinc-100 dark:bg-zinc-950">
                <ProductScreenshot />
              </div>
            </motion.div>
          </motion.div>
        </motion.div>
      </section>
    </>
  );
}


