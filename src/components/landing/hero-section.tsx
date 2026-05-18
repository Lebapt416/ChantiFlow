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
          <div className="w-24 h-24 mx-auto mb-4 rounded bg-orange flex items-center justify-center">
            <svg className="w-12 h-12 text-paper" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
            </svg>
          </div>
          <p className="text-sm text-ink-3">Capture d&apos;écran du planning</p>
        </div>
      </div>
    );
  }

  return (
    <Image
      src="/capture-planning.png"
      alt="Capture d'écran du planning ChantiFlow - Interface de planification intelligente avec drag & drop"
      fill
      className="object-contain p-4"
      priority
      onError={() => setImageError(true)}
      sizes="(max-width: 768px) 100vw, (max-width: 1200px) 90vw, 1200px"
      quality={85}
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
  const [mounted] = useState(true);

  // Rendu statique initial pour le FCP
  if (!mounted) {
    return (
      <>
        <header className="sticky top-0 z-50 border-b border-rule-soft bg-paper">
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
                <span className="text-xl font-bold text-ink">
                  ChantiFlow
                </span>
              </div>
              <nav className="flex items-center gap-6">
                <Link
                  href="/login"
                  className="text-sm font-medium text-ink-2 transition hover:text-ink"
                >
                  Connexion
                </Link>
                <Link
                  href="/login"
                  className="rounded bg-ink px-5 py-2 text-sm font-semibold text-paper transition hover:bg-rule"
                >
                  Commencer
                </Link>
              </nav>
            </div>
          </div>
        </header>

        <section className="relative mx-auto max-w-7xl px-6 py-20 overflow-hidden">
          <div className="mx-auto max-w-4xl text-center">
            <span className="inline-block rounded bg-paper-2 px-4 py-1.5 text-sm font-semibold text-ink border border-rule-soft">
              Logiciel de gestion BTP
            </span>
            <h1 className="relative mt-6 text-5xl font-extrabold tracking-tight sm:text-6xl lg:text-7xl text-ink">
              Fini les retards de chantier.
            </h1>
            <p className="mt-6 text-xl leading-8 text-ink-2 max-w-2xl mx-auto font-medium">
              Le seul planning qui se met à jour tout seul quand une tâche prend du retard. Plus d&apos;appels le soir, plus de papier.
            </p>
            <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/login"
                className="rounded bg-orange px-8 py-4 text-lg font-bold text-paper transition-colors duration-200 hover:bg-orange-dark"
              >
                Lancer mon 1er chantier
              </Link>
              <Link
                href="/contact"
                className="rounded border border-rule px-8 py-4 text-lg font-semibold text-ink transition-colors duration-200 hover:bg-paper-2"
              >
                Nous contacter
              </Link>
            </div>
            <div className="mt-16 relative">
              <div className="relative mx-auto max-w-6xl rounded border border-rule bg-paper overflow-hidden">
                <div
                  className="relative bg-paper-2"
                  style={{
                    aspectRatio: "1353 / 768",
                    minHeight: "400px",
                  }}
                >
                  <ProductScreenshot />
                </div>
              </div>
            </div>
          </div>
        </section>
      </>
    );
  }

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-rule-soft bg-paper">
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
              <span className="text-xl font-bold text-ink">
                ChantiFlow
              </span>
            </div>
            <nav className="flex items-center gap-6">
              <Link
                href="/login"
                className="text-sm font-medium text-ink-2 transition hover:text-ink"
              >
                Connexion
              </Link>
              <Link
                href="/login"
                className="rounded bg-ink px-5 py-2 text-sm font-semibold text-paper transition hover:bg-rule"
              >
                Commencer
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <section className="relative mx-auto max-w-7xl px-6 py-20 overflow-hidden">
        <motion.div
          className="mx-auto max-w-4xl text-center"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Badge */}
          <motion.div variants={itemVariants}>
            <span className="inline-block rounded bg-paper-2 px-4 py-1.5 text-sm font-semibold text-ink border border-rule-soft">
              Logiciel de gestion BTP
            </span>
          </motion.div>

          {/* Titre */}
          <motion.h1
            variants={itemVariants}
            className="relative mt-6 text-5xl font-extrabold tracking-tight sm:text-6xl lg:text-7xl text-ink"
          >
            Fini les retards de chantier.
          </motion.h1>

          {/* Sous-titre */}
          <motion.p
            variants={itemVariants}
            className="mt-6 text-xl leading-8 text-ink-2 max-w-2xl mx-auto font-medium"
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
              className="rounded bg-orange px-8 py-4 text-lg font-bold text-paper transition-colors duration-200 hover:bg-orange-dark"
            >
              <span className="relative">Lancer mon 1er chantier</span>
            </Link>
            <Link
              href="/contact"
              className="rounded border border-rule px-8 py-4 text-lg font-semibold text-ink transition-colors duration-200 hover:bg-paper-2"
            >
              <span className="relative">Nous contacter</span>
            </Link>
          </motion.div>

          {/* Product screenshot */}
          <motion.div
            variants={itemVariants}
            className="mt-16 relative"
          >
            <div className="relative mx-auto max-w-6xl rounded border border-rule bg-paper overflow-hidden">
              <div
                className="relative bg-paper-2"
                style={{
                  aspectRatio: "1353 / 768",
                  minHeight: "400px",
                }}
              >
                <ProductScreenshot />
              </div>
            </div>
          </motion.div>
        </motion.div>
      </section>
    </>
  );
}
