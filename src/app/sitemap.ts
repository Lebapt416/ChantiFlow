import type { MetadataRoute } from "next";

const baseUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? "https://www.chantiflow.com";

const marketingRoutes = [
  "",
  "/landing",
  "/login",
  "/worker/login",
  "/mentions-legales",
  "/politique-confidentialite",
];

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date();

  return marketingRoutes.map((path) => ({
    url: `${baseUrl}${path}`,
    lastModified,
    changeFrequency: path === "/landing" || path === "" ? "daily" : "weekly",
    priority: path === "/landing" || path === "" ? 1 : 0.6,
  }));
}

