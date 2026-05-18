import { AuthFormAPI } from './auth-form-api';

export const metadata = {
  title: 'Connexion | ChantiFlow',
};

export default function LoginPage({
  searchParams,
}: {
  searchParams: { reset?: string };
}) {
  return (
    <div className="flex min-h-screen">

      {/* Left column — editorial quote, hidden on mobile */}
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-between bg-paper-2 p-16 border-r border-rule">

        {/* Logo */}
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 bg-ink text-paper flex items-center justify-center font-mono text-base font-medium"
            style={{ transform: 'rotate(-3deg)' }}
          >
            C
          </div>
          <span
            className="font-serif text-xl font-semibold text-ink"
            style={{ letterSpacing: '-0.02em', fontVariationSettings: '"opsz" 144, "SOFT" 50' }}
          >
            ChantiFlow
          </span>
        </div>

        {/* Editorial quote */}
        <div>
          <div className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-6">
            Chapitre · Accès
          </div>
          <blockquote
            className="font-serif text-[42px] font-normal leading-[1.05] tracking-tight text-ink"
            style={{ fontVariationSettings: '"opsz" 144, "SOFT" 30' }}
          >
            Le chantier ne s&apos;arrête pas.<br />
            <em
              className="italic font-light text-orange"
              style={{ fontVariationSettings: '"opsz" 144, "SOFT" 100' }}
            >
              Votre planning
            </em>{' '}
            non plus.
          </blockquote>
          <p className="mt-6 text-[15px] text-ink-2 max-w-sm leading-relaxed">
            Gérez vos chantiers, planifiez vos équipes et suivez l&apos;avancement — depuis le bureau ou le terrain.
          </p>
        </div>

        {/* Bottom mono label */}
        <div className="font-mono text-[10px] uppercase tracking-widest text-ink-3">
          Hébergement UE · RGPD ✓ · Données chiffrées
        </div>
      </div>

      {/* Right column — form */}
      <div className="flex w-full flex-col justify-center bg-paper px-8 py-12 lg:w-1/2 lg:px-20">

        {/* Mobile logo */}
        <div className="mb-10 flex items-center gap-3 lg:hidden">
          <div
            className="w-8 h-8 bg-ink text-paper flex items-center justify-center font-mono text-base font-medium"
            style={{ transform: 'rotate(-3deg)' }}
          >
            C
          </div>
          <span className="font-serif text-xl font-semibold text-ink">ChantiFlow</span>
        </div>

        <div className="w-full max-w-sm">
          <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-2">
            Espace chef de chantier
          </p>
          <h1
            className="font-serif text-[32px] font-normal text-ink mb-1"
            style={{ fontVariationSettings: '"opsz" 60, "SOFT" 30' }}
          >
            Connexion
          </h1>
          <p className="text-[15px] text-ink-2 mb-8">
            Connecte-toi ou crée un compte pour commencer.
          </p>

          {/* Reset success banner */}
          {searchParams?.reset === 'success' && (
            <div className="mb-6 border border-green bg-paper-2 px-4 py-3 font-mono text-[12px] tracking-wide text-green">
              ✓ Mot de passe réinitialisé avec succès.
            </div>
          )}

          <AuthFormAPI />
        </div>
      </div>
    </div>
  );
}
