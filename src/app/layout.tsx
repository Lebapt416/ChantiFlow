import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { PwaRegister } from "@/components/pwa-register";
import { AuthProvider } from "@/components/auth-provider";
import { OfflineIndicator } from "@/components/offline-indicator";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
  preload: true,
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
  preload: false, // Moins prioritaire
});

const appBaseUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? "https://www.chantiflow.com";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "white" },
    { media: "(prefers-color-scheme: dark)", color: "#18181b" },
  ],
};

export const metadata: Metadata = {
  metadataBase: new URL("https://chantiflow.com"),
  title: {
    default: "ChantiFlow - Logiciel de Gestion de Chantier & Planning IA",
    template: "%s | ChantiFlow",
  },
  description: "Logiciel SaaS pour planifier, suivre et automatiser la gestion de vos chantiers BTP.",
  keywords: [
    "chantier",
    "logiciel btp",
    "gestion de chantier",
    "planning chantier",
    "suivi chantier",
    "QR code chantier",
    "application btp",
    "rapport chantier",
  ],
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    url: appBaseUrl,
    siteName: "ChantiFlow",
    title: "ChantiFlow - Logiciel de gestion de chantiers BTP",
    description:
      "Automatisez votre planning, vos équipes et vos rapports de chantier grâce à l'IA. Solution SaaS dédiée aux entreprises du BTP.",
    images: [
      {
        url: `${appBaseUrl}/og-image.png`,
        width: 1200,
        height: 630,
        alt: "Interface ChantiFlow",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    site: "@chantiflow",
    title: "ChantiFlow - Gestion de chantiers",
    description:
      "Planificateur BTP avec IA, QR codes équipes et rapports automatiques pour vos chantiers.",
    images: [`${appBaseUrl}/og-image.png`],
  },
  category: "technology",
  icons: {
    icon: [
      { url: "/favicon.svg", rel: "icon", type: "image/svg+xml" },
      { url: "/favicon.ico", rel: "icon", sizes: "16x16 32x32 48x48" },
      { url: "/logo.svg", sizes: "any", rel: "icon" },
    ],
    apple: [{ url: "/apple-touch-icon.png", sizes: "180x180" }],
  },
  other: {
    "format-detection": "telephone=no",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr">
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="apple-mobile-web-app-title" content="ChantiFlow" />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[var(--background)] text-[var(--foreground)]`}
      >
        <AuthProvider>
          {children}
          <PwaRegister />
          <OfflineIndicator />
          <SpeedInsights />
          {/* Debug temporaire pour identifier les reloads */}
          {process.env.NODE_ENV === 'development' && (
            <script
              dangerouslySetInnerHTML={{
                __html: `
                  (function() {
                    let reloadCount = 0;
                    const originalReload = window.location.reload;
                    window.location.reload = function() {
                      reloadCount++;
                      console.error('[DEBUG] 🔴 RELOAD #' + reloadCount, new Error().stack);
                      return originalReload.apply(this, arguments);
                    };
                    window.addEventListener('beforeunload', function() {
                      console.warn('[DEBUG] ⚠️ BEFOREUNLOAD triggered');
                    });
                    console.log('[DEBUG] Script de diagnostic chargé');
                  })();
                `,
              }}
            />
          )}
        </AuthProvider>
      </body>
    </html>
  );
}
