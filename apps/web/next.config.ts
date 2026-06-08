import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // API base URL is read at runtime from NEXT_PUBLIC_API_BASE_URL (see lib/api.ts).
};

export default nextConfig;
