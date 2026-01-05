'use client';

import Image from "next/image";
import Link from "next/link";
import { useState, useEffect } from "react";
import dynamic from "next/dynamic";

// Lazy load framer-motion uniquement après le montage
const AnimatedBackground = dynamic(() => import('./hero-animated-background').then(mod => ({ default: mod.AnimatedBackground })), {
  ssr: false,
});

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
      alt="Capture d'écran du planning ChantiFlow - Interface de planification intelligente avec drag & drop"
      fill
      className="object-contain p-4"
      priority
      onError={() => setImageError(true)}
      sizes="(max-width: 768px) 100vw, (max-width: 1200px) 90vw, 1200px"
      quality={75}
      loading="eager"
      fetchPriority="high"
    />
  );
}

export function HeroSection() {
  const [mounted, setMounted] = useState(false);

  // Ne charger les animations qu'après le montage pour ne pas bloquer le FCP
  useEffect(() => {
    // Utiliser requestIdleCallback pour différer le chargement des animations
    if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
      requestIdleCallback(() => {
        setMounted(true);
      }, { timeout: 2000 });
    } else {
      // Fallback pour navigateurs sans requestIdleCallback
      setTimeout(() => setMounted(true), 100);
    }
  }, []);

  // Rendu statique initial pour le FCP (critique pour le LCP)
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
                fetchPriority="high"
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
        {/* Fond animé - lazy load après montage */}
        <div className="absolute inset-0 -z-10 overflow-hidden">
          {mounted ? (
            <AnimatedBackground />
          ) : (
            // Version statique pour le FCP
            <>
              <div className="absolute left-1/4 top-1/4 w-[600px] h-[600px] bg-emerald-500/20 dark:bg-emerald-400/20 rounded-full blur-3xl" />
              <div className="absolute right-1/4 top-1/3 w-[500px] h-[500px] bg-blue-500/20 dark:bg-blue-400/20 rounded-full blur-3xl" />
              <div className="absolute left-1/2 bottom-1/4 w-[550px] h-[550px] bg-purple-500/20 dark:bg-purple-400/20 rounded-full blur-3xl" />
            </>
          )}
        </div>

        <div className="mx-auto max-w-4xl text-center">
          {/* Badge */}
          <div>
            <span className="inline-block rounded-full bg-emerald-100 dark:bg-emerald-900/30 px-4 py-1.5 text-sm font-semibold text-emerald-700 dark:text-emerald-400 border border-emerald-200/50 dark:border-emerald-800/50">
              Logiciel de gestion BTP
            </span>
          </div>

          {/* Titre avec gradient */}
          <h1 className="relative mt-6 text-5xl font-extrabold tracking-tight sm:text-6xl lg:text-7xl">
            <span className="bg-gradient-to-r from-emerald-600 via-emerald-500 to-cyan-500 bg-clip-text text-transparent dark:from-emerald-400 dark:via-emerald-300 dark:to-cyan-400">
              Fini les retards de chantier.
            </span>
          </h1>

          {/* Sous-titre */}
          <p className="mt-6 text-xl leading-8 text-zinc-700 dark:text-zinc-300 max-w-2xl mx-auto font-medium">
            Le seul planning qui se met à jour tout seul quand une tâche prend du retard. Plus d&apos;appels le soir, plus de papier.
          </p>
          
          {/* Boutons */}
          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/login"
              className="group relative overflow-hidden rounded-xl bg-gradient-to-r from-emerald-600 to-emerald-500 px-8 py-4 text-lg font-bold text-white shadow-lg shadow-emerald-500/25 transition-all duration-300 hover:scale-105 hover:shadow-emerald-500/40 hover:from-emerald-500 hover:to-emerald-400 backdrop-blur-md border border-white/20 will-change-transform"
            >
              <span className="relative">Lancer mon 1er chantier</span>
            </Link>
            <Link
              href="/contact"
              className="group relative overflow-hidden rounded-xl border-2 border-white/20 dark:border-zinc-700/50 bg-white/30 dark:bg-zinc-900/30 backdrop-blur-md px-8 py-4 text-lg font-semibold text-zinc-900 dark:text-white shadow-lg transition-all duration-300 hover:scale-105 hover:bg-white/40 dark:hover:bg-zinc-800/40 hover:border-white/30 will-change-transform"
            >
              <span className="relative">Nous contacter</span>
            </Link>
          </div>

          {/* Product screenshot */}
          <div className="mt-16 relative">
            <div className="relative mx-auto max-w-6xl rounded-2xl border border-white/20 dark:border-zinc-800/50 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md shadow-2xl shadow-emerald-500/20 overflow-hidden">
              <div
                className="relative bg-zinc-100 dark:bg-zinc-950"
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

