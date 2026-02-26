/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy API calls to the backend in production (avoids CORS entirely)
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL;
    if (!backendUrl) return [];          // dev: frontend calls backend directly
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;

