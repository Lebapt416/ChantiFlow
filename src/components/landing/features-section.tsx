'use client';

export function FeaturesSection() {
  const features = [
    {
      ix: '01 / Réactif',
      title: <>Le planning se <em className="font-light italic text-orange">recalcule seul</em> quand ça déraille.</>,
      desc: 'Un retard signalé sur le terrain → toutes les tâches en aval se replanifient automatiquement. Sous-traitants, livraisons et équipes sont prévenus dans la foulée.',
      tag: '⏱ Gain moyen — 3,4 h / sem.',
    },
    {
      ix: '02 / Terrain',
      title: <>QR codes pour les <em className="font-light italic text-orange">ouvriers</em>, pas pour les DSI.</>,
      desc: "Les équipes pointent en scannant un QR à l'entrée du chantier. Pas d'app à installer, pas de mot de passe à retenir. Compatible avec les téléphones d'entrée de gamme.",
      tag: '📱 0 install · iOS & Android',
    },
    {
      ix: '03 / Discret',
      title: <>Rapports clients <em className="font-light italic text-orange">générés</em>, pas écrits.</>,
      desc: "Photos prises sur place + tâches du jour → un rapport hebdo PDF propre, signé numériquement et envoyé au maître d'ouvrage. Le vendredi soir, c'est fini.",
      tag: '📄 PDF · signature électronique',
    },
  ];

  return (
    <section id="features" className="border-t border-rule py-20">
      <div className="mx-auto max-w-7xl px-8">
        {/* Chapter header */}
        <div className="grid grid-cols-[auto_1fr] gap-20 items-end pb-16 border-b border-rule-soft mb-16">
          <div className="font-mono text-[12px] text-ink-2 tracking-widest">
            <span className="block font-serif text-[96px] italic font-light text-ink leading-none mb-2" style={{fontVariationSettings: '"opsz" 144, "SOFT" 100'}}>02</span>
            Chapitre · Fonctionnalités
          </div>
          <div>
            <h2 className="font-serif font-normal text-ink leading-[1.02] tracking-tight" style={{fontSize: 'clamp(36px, 4vw, 56px)', fontVariationSettings: '"opsz" 144, "SOFT" 30'}}>
              Trois choses qu&apos;aucun<br/>
              autre logiciel ne fait <em className="italic font-light text-orange">bien.</em>
            </h2>
            <p className="text-[17px] text-ink-2 mt-5 max-w-[540px] leading-relaxed">
              On a passé six mois sur le terrain avec des conducteurs de travaux, des artisans et des chefs d&apos;entreprise BTP.
            </p>
          </div>
        </div>

        {/* Feature grid */}
        <div className="grid grid-cols-3 border-t border-l border-rule-soft">
          {features.map((f, i) => (
            <div key={i} className="p-10 border-r border-b border-rule-soft bg-paper hover:bg-paper-2 transition-colors duration-200">
              <div className="font-mono text-[11px] tracking-widest text-ink-2 mb-20">{f.ix}</div>
              <h3 className="font-serif text-[26px] font-medium leading-tight mb-3" style={{fontVariationSettings: '"opsz" 60, "SOFT" 30'}}>{f.title}</h3>
              <p className="text-[15px] text-ink-2 leading-relaxed">{f.desc}</p>
              <span className="inline-block mt-4 font-mono text-[10px] uppercase tracking-widest px-2 py-1 border border-rule-soft text-ink-2">{f.tag}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
