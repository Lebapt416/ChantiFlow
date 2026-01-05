import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '*.supabase.co',
        pathname: '/storage/v1/object/public/**',
      },
    ],
  },
  // Compression activée pour réduire la taille des réponses
  compress: true,
  // Forcer la compilation des Server Actions
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
  // Désactiver le cache agressif qui pourrait causer des problèmes
  onDemandEntries: {
    maxInactiveAge: 25 * 1000,
    pagesBufferLength: 2,
  },
};

export default nextConfig;
