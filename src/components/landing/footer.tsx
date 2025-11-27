import Image from "next/image";
import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mx-auto max-w-7xl px-6 py-12">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
          <div className="col-span-1 md:col-span-2">
            <div className="mb-4 flex items-center gap-3">
              <Image
                src="/logo.svg"
                alt="ChantiFlow"
                width={32}
                height={32}
                className="h-8 w-8"
              />
              <span className="text-lg font-bold text-zinc-900 dark:text-white">
                ChantiFlow
              </span>
            </div>
            <p className="max-w-md text-sm text-zinc-600 dark:text-zinc-400">
              La gestion de chantier simplifiée, assistée par l&apos;intelligence artificielle.
            </p>
          </div>

          <div>
            <h3 className="mb-4 text-sm font-semibold text-zinc-900 dark:text-white">Légal</h3>
            <ul className="space-y-2">
              <li>
                <Link
                  href="/mentions-legales"
                  className="text-sm text-zinc-600 transition hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
                >
                  Mentions légales
                </Link>
              </li>
              <li>
                <Link
                  href="/politique-confidentialite"
                  className="text-sm text-zinc-600 transition hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
                >
                  Politique de confidentialité
                </Link>
              </li>
              <li>
                <Link
                  href="/rgpd"
                  className="text-sm text-zinc-600 transition hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
                >
                  RGPD
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="mb-4 text-sm font-semibold text-zinc-900 dark:text-white">Contact</h3>
            <ul className="space-y-2">
              <li>
                <a
                  href="mailto:contact@chantiflow.com"
                  className="text-sm text-zinc-600 transition hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
                >
                  contact@chantiflow.com
                </a>
              </li>
              <li>
                <a
                  href="mailto:admin@chantiflow.com"
                  className="text-sm text-zinc-600 transition hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
                >
                  admin@chantiflow.com
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 border-t border-zinc-200 pt-8 dark:border-zinc-800">
          <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
            <p className="text-sm text-zinc-600 dark:text-zinc-400">© 2025 ChantiFlow. Tous droits réservés.</p>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">ChantiFlow SAS</p>
          </div>
        </div>
      </div>
    </footer>
  );
}


