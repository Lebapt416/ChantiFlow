import Image from "next/image";
import Link from "next/link";

export function HeroSection() {
  return (
    <>
      <header className="border-b border-zinc-200 dark:border-zinc-800">
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

      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="text-sm font-semibold uppercase tracking-[0.4em] text-emerald-600 dark:text-emerald-400">
            Logiciel de gestion BTP
          </h2>
          <h1 className="text-5xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-6xl">
            Logiciel de Suivi de Chantier&nbsp;: gérez vos chantiers{" "}
            <span className="text-emerald-600 dark:text-emerald-400">30% plus vite</span>{" "}
            grâce à l&apos;IA
          </h1>
          <p className="mt-6 text-lg leading-8 text-zinc-600 dark:text-zinc-400">
            ChantiFlow automatise votre planning, centralise vos équipes, génère vos rapports, et
            suit l&apos;avancement de vos chantiers en temps réel — sans Excel, sans appels
            inutiles.
          </p>
          <div className="mt-10 flex flex-col items-center justify-center gap-4">
            <Link
              href="/login"
              className="rounded-full bg-black px-8 py-4 text-lg font-bold text-white shadow-xl transition hover:bg-zinc-800 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
            >
              Démarrer gratuitement
            </Link>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">✓ Compatible smartphone et QR codes de chantier</p>
          </div>
        </div>
      </section>
    </>
  );
}


