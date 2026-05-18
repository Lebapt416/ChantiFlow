import Link from 'next/link';

export function Footer() {
  return (
    <footer className="bg-ink text-paper py-6 font-mono text-[11px] tracking-widest uppercase border-t border-rule">
      <div className="mx-auto max-w-7xl px-8">
        <div className="flex items-center justify-between gap-8">
          <span><strong className="font-medium">ChantiFlow</strong> · Normandie, France</span>
          <div className="flex items-center gap-6 opacity-65">
            <Link href="/mentions-legales" className="hover:opacity-100 transition-opacity">Mentions légales</Link>
            <Link href="/politique-confidentialite" className="hover:opacity-100 transition-opacity">Confidentialité</Link>
            <Link href="/rgpd" className="hover:opacity-100 transition-opacity">RGPD</Link>
            <Link href="/contact" className="hover:opacity-100 transition-opacity">Contact</Link>
          </div>
          <span className="opacity-65">Hébergement UE · RGPD ✓</span>
        </div>
      </div>
    </footer>
  );
}
