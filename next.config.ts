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
  // Forcer la revalidation pour éviter les problèmes de cache
  experimental: {
    staleTimes: {
      dynamic: 0,
      static: 0,
    },
  },
};

export default nextConfig;
