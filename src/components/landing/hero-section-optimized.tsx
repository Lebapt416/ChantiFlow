'use client';

import Link from 'next/link';

export function HeroSection() {
  return (
    <>
      {/* Sticky header */}
      <header className="sticky top-0 z-50 bg-paper border-b border-rule">
        <div className="mx-auto max-w-7xl px-8 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 bg-ink text-paper flex items-center justify-center font-mono text-lg font-medium" style={{transform: 'rotate(-3deg)'}}>
                C
              </div>
              <span className="font-serif text-2xl font-semibold text-ink" style={{letterSpacing: '-0.02em', fontVariationSettings: '"opsz" 144, "SOFT" 50'}}>
                ChantiFlow<sup className="font-mono text-ink-2 ml-1 tracking-widest uppercase" style={{verticalAlign: 'super', fontSize: '10px'}}>FR</sup>
              </span>
            </div>

            {/* Nav */}
            <nav className="flex items-center gap-7">
              <Link href="#features" className="text-sm font-medium text-ink hover:underline underline-offset-4 decoration-1 transition-colors">Produit</Link>
              <Link href="#pricing" className="text-sm font-medium text-ink hover:underline underline-offset-4 decoration-1 transition-colors">Tarifs</Link>
              <Link href="/contact" className="text-sm font-medium text-ink hover:underline underline-offset-4 decoration-1 transition-colors">Contact</Link>
              <Link href="/login" className="inline-flex items-center gap-2 border border-ink bg-ink px-5 py-2.5 text-sm font-medium text-paper transition-colors hover:bg-orange hover:border-orange">
                Connexion <span>→</span>
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero section */}
      <section className="py-20 border-b border-rule">
        <div className="mx-auto max-w-7xl px-8">
          {/* Meta row */}
          <div className="flex items-center justify-between border-b border-rule-soft pb-7 mb-16 font-mono text-[11px] uppercase tracking-widest text-ink-2">
            <span>N° 001 / Planification de chantier</span>
            <span>── Édition Bêta · Printemps 2026 ──</span>
            <span>Normandie → France entière</span>
          </div>

          {/* Hero grid */}
          <div className="grid grid-cols-[1fr_300px] gap-16 items-end">
            {/* Left: H1 */}
            <div>
              <h1 className="font-serif font-normal text-ink" style={{fontSize: 'clamp(56px, 8vw, 112px)', lineHeight: 0.94, letterSpacing: '-0.035em', fontVariationSettings: '"opsz" 144, "SOFT" 30'}}>
                Fini les<br/>
                <span style={{fontStyle: 'italic', fontWeight: 300, fontVariationSettings: '"opsz" 144, "SOFT" 100', color: 'var(--orange)', borderBottom: '0.08em solid', borderColor: 'var(--orange)'}}>retards</span><br/>
                de chantier.
                <span style={{display: 'block', fontSize: '0.42em', fontWeight: 500, letterSpacing: '-0.02em', color: 'var(--ink-2)', marginTop: '0.4em', fontVariationSettings: '"opsz" 60, "SOFT" 30', fontStyle: 'normal'}}>— Le planning qui se met à jour tout seul.</span>
              </h1>
            </div>

            {/* Right: aside */}
            <aside className="border-l border-rule pl-7">
              <p className="text-[17px] text-ink leading-relaxed">
                Une tâche prend du retard ?{' '}
                <strong className="bg-orange text-paper px-1">Tout le planning s&apos;ajuste</strong>
                , les équipes sont prévenues.
              </p>
              <div className="flex flex-col gap-2.5 mt-6">
                <Link
                  href="/login"
                  className="inline-flex items-center justify-between gap-3 bg-orange text-paper border border-orange px-5 py-3 text-sm font-medium transition-colors hover:bg-orange-dark"
                >
                  Lancer mon 1er chantier
                  <span className="font-mono text-[11px] opacity-60">01 →</span>
                </Link>
                <Link
                  href="/contact"
                  className="inline-flex items-center justify-between gap-3 bg-transparent text-ink border border-rule px-5 py-3 text-sm font-medium transition-colors hover:bg-ink hover:text-paper"
                >
                  Voir une démo (3 min)
                  <span className="font-mono text-[11px] opacity-60">02 →</span>
                </Link>
              </div>
            </aside>
          </div>

          {/* Planning showcase */}
          <div className="mt-16 relative">
            <div
              className="font-mono text-[10px] tracking-widest text-ink-2 mb-1 uppercase"
            >
              ECRAN.PLANNING / V.2026 / CAPT-001
            </div>
            <div className="bg-ink p-3 border border-ink">
              <svg viewBox="0 0 1353 600" xmlns="http://www.w3.org/2000/svg" className="block w-full" style={{aspectRatio: '1353/600', background: '#FAFAF7'}}>
                {/* Top bar */}
                <rect x="0" y="0" width="1353" height="48" fill="#161512"/>
                <circle cx="20" cy="24" r="5" fill="#E85D1F"/>
                <circle cx="38" cy="24" r="5" fill="#B58105"/>
                <circle cx="56" cy="24" r="5" fill="#2F5D3A"/>
                <text x="90" y="29" fill="#F1EBDF" fontFamily="JetBrains Mono,monospace" fontSize="12" letterSpacing="1">CHANTIFLOW / PLANNING — Q2 2026</text>
                <text x="1280" y="29" fill="#F1EBDF" opacity="0.6" fontFamily="JetBrains Mono,monospace" fontSize="12">⌘K</text>
                {/* Left sidebar */}
                <rect x="0" y="48" width="220" height="552" fill="#E8E1D2"/>
                <text x="20" y="80" fill="#5A554B" fontFamily="JetBrains Mono,monospace" fontSize="10" letterSpacing="1.5">CHANTIERS · 12</text>
                <g fontFamily="Bricolage Grotesque,sans-serif" fontSize="13" fill="#161512">
                  <rect x="14" y="92" width="192" height="32" fill="#161512"/>
                  <text x="24" y="113" fill="#F1EBDF">▸ Résidence Les Tilleuls</text>
                  <text x="24" y="143">▸ Atelier Garnier</text>
                  <text x="24" y="171">▸ Pavillon Caen-Nord</text>
                  <text x="24" y="199">▸ Cabinet médical Hérouville</text>
                  <text x="24" y="227">▸ Halle métallique Bayeux</text>
                  <text x="24" y="255">▸ Réhab. logements ANRU</text>
                  <text x="24" y="283">▸ Showroom Lemonnier</text>
                </g>
                <line x1="220" y1="48" x2="220" y2="600" stroke="#3A352D" strokeWidth="0.5"/>
                {/* Main planning area */}
                <text x="244" y="80" fill="#161512" fontFamily="Fraunces,serif" fontSize="22" fontStyle="italic">Résidence Les Tilleuls</text>
                <text x="244" y="100" fill="#5A554B" fontFamily="JetBrains Mono,monospace" fontSize="10" letterSpacing="1">RUE DES TILLEULS · CAEN · LOT 04/12</text>
                {/* Date headers */}
                <g fontFamily="JetBrains Mono,monospace" fontSize="10" fill="#5A554B" letterSpacing="0.5">
                  <text x="260" y="148">L 18</text><text x="360" y="148">M 19</text><text x="460" y="148">M 20</text>
                  <text x="560" y="148">J 21</text><text x="660" y="148">V 22</text>
                  <text x="760" y="148" opacity="0.4">S 23</text><text x="860" y="148" opacity="0.4">D 24</text>
                  <text x="960" y="148">L 25</text><text x="1060" y="148">M 26</text>
                  <text x="1160" y="148">M 27</text><text x="1260" y="148">J 28</text>
                </g>
                {/* Grid lines */}
                <g stroke="#C9BFAC" strokeWidth="0.5">
                  <line x1="244" y1="160" x2="1340" y2="160"/><line x1="244" y1="220" x2="1340" y2="220"/>
                  <line x1="244" y1="280" x2="1340" y2="280"/><line x1="244" y1="340" x2="1340" y2="340"/>
                  <line x1="244" y1="400" x2="1340" y2="400"/><line x1="244" y1="460" x2="1340" y2="460"/>
                  <line x1="244" y1="520" x2="1340" y2="520"/>
                </g>
                <g stroke="#C9BFAC" strokeWidth="0.5" strokeDasharray="2,3">
                  <line x1="360" y1="160" x2="360" y2="540"/><line x1="460" y1="160" x2="460" y2="540"/>
                  <line x1="560" y1="160" x2="560" y2="540"/><line x1="660" y1="160" x2="660" y2="540"/>
                  <line x1="760" y1="160" x2="760" y2="540"/><line x1="860" y1="160" x2="860" y2="540"/>
                  <line x1="960" y1="160" x2="960" y2="540"/><line x1="1060" y1="160" x2="1060" y2="540"/>
                  <line x1="1160" y1="160" x2="1160" y2="540"/><line x1="1260" y1="160" x2="1260" y2="540"/>
                </g>
                {/* Row labels */}
                <g fontFamily="Bricolage Grotesque,sans-serif" fontSize="12" fill="#5A554B">
                  <text x="244" y="194">Gros œuvre</text><text x="244" y="254">Charpente</text>
                  <text x="244" y="314">Plomberie</text><text x="244" y="374">Électricité</text>
                  <text x="244" y="434">Cloisons / Plâtrerie</text><text x="244" y="494">Peinture / Finitions</text>
                </g>
                {/* Task bars */}
                <rect x="260" y="174" width="280" height="28" fill="#2F5D3A" rx="2"/>
                <text x="272" y="193" fill="#F1EBDF" fontFamily="Bricolage Grotesque,sans-serif" fontSize="12" fontWeight="500">✓ Dalle + élévations · Équipe Martinez</text>
                <rect x="540" y="234" width="320" height="28" fill="#E85D1F" rx="2"/>
                <text x="552" y="253" fill="#F1EBDF" fontFamily="Bricolage Grotesque,sans-serif" fontSize="12" fontWeight="500">⚠ Charpente bois · +2j de retard détecté</text>
                <rect x="860" y="294" width="180" height="28" fill="#1E3A8A" rx="2"/>
                <text x="872" y="313" fill="#F1EBDF" fontFamily="Bricolage Grotesque,sans-serif" fontSize="12" fontWeight="500">Plomberie · Dubois &amp; Fils</text>
                <rect x="960" y="354" width="220" height="28" fill="#1E3A8A" rx="2"/>
                <text x="972" y="373" fill="#F1EBDF" fontFamily="Bricolage Grotesque,sans-serif" fontSize="12" fontWeight="500">Électricité · Sous-traitance EDF</text>
                <rect x="1080" y="414" width="200" height="28" fill="none" stroke="#1E3A8A" strokeWidth="1.5" strokeDasharray="3,3" rx="2"/>
                <text x="1092" y="433" fill="#1E3A8A" fontFamily="Bricolage Grotesque,sans-serif" fontSize="12" fontWeight="500">Cloisons (replanifié auto)</text>
                <rect x="1200" y="474" width="120" height="28" fill="none" stroke="#5A554B" strokeWidth="1" rx="2"/>
                <text x="1212" y="493" fill="#5A554B" fontFamily="Bricolage Grotesque,sans-serif" fontSize="12">Peinture · à planifier</text>
                {/* Today line */}
                <line x1="565" y1="160" x2="565" y2="540" stroke="#E85D1F" strokeWidth="1.5"/>
                <rect x="540" y="558" width="50" height="18" fill="#E85D1F"/>
                <text x="550" y="571" fill="#F1EBDF" fontFamily="JetBrains Mono,monospace" fontSize="10" letterSpacing="1">AUJ.</text>
                {/* Legend */}
                <g transform="translate(900,70)">
                  <rect x="0" y="0" width="12" height="12" fill="#2F5D3A"/>
                  <text x="18" y="10" fill="#5A554B" fontFamily="JetBrains Mono,monospace" fontSize="10">VALIDÉ</text>
                  <rect x="90" y="0" width="12" height="12" fill="#E85D1F"/>
                  <text x="108" y="10" fill="#5A554B" fontFamily="JetBrains Mono,monospace" fontSize="10">RETARD</text>
                  <rect x="180" y="0" width="12" height="12" fill="#1E3A8A"/>
                  <text x="198" y="10" fill="#5A554B" fontFamily="JetBrains Mono,monospace" fontSize="10">PRÉVU</text>
                  <rect x="270" y="0" width="12" height="12" fill="none" stroke="#1E3A8A" strokeDasharray="2,2"/>
                  <text x="288" y="10" fill="#5A554B" fontFamily="JetBrains Mono,monospace" fontSize="10">RECALC.</text>
                </g>
              </svg>
            </div>
          </div>
        </div>
      </section>

      {/* Stats strip */}
      <section className="border-b border-rule py-10">
        <div className="mx-auto max-w-7xl px-8">
          <div className="grid grid-cols-4 divide-x divide-rule-soft">
            <div className="px-7 first:pl-0">
              <div className="font-serif text-[56px] leading-none tracking-[-0.04em] text-ink">−42<span className="text-[0.5em] text-orange italic align-super">%</span></div>
              <div className="mt-3.5 font-mono text-[10px] uppercase tracking-widest text-ink-2">Retards constatés · Bêta n = 18</div>
            </div>
            <div className="px-7">
              <div className="font-serif text-[56px] leading-none tracking-[-0.04em] text-ink">3,4<span className="text-[0.5em] text-orange italic align-super">h</span></div>
              <div className="mt-3.5 font-mono text-[10px] uppercase tracking-widest text-ink-2">Économisées / semaine / conducteur</div>
            </div>
            <div className="px-7">
              <div className="font-serif text-[56px] leading-none tracking-[-0.04em] text-ink">100<span className="text-[0.5em] text-orange italic align-super">%</span></div>
              <div className="mt-3.5 font-mono text-[10px] uppercase tracking-widest text-ink-2">Des ouvriers bêta utilisent le QR</div>
            </div>
            <div className="px-7">
              <div className="font-serif text-[56px] leading-none tracking-[-0.04em] text-ink">48<span className="text-[0.5em] text-orange italic align-super">h</span></div>
              <div className="mt-3.5 font-mono text-[10px] uppercase tracking-widest text-ink-2">Délai d&apos;onboarding moyen · premier chantier</div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
