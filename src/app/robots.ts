import type { MetadataRoute } from "next";

const baseUrl = process.env.NEXT_PUBLIC_APP_BASE_URL ?? "https://www.chantiflow.com";
const baseHost = new URL(baseUrl).host;

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: "*",
      allow: "/",
      disallow: ["/dashboard", "/tasks", "/site", "/api"],
    },
    sitemap: [`${baseUrl}/sitemap.xml`],
    host: baseHost,
  };
}

