/** @type {import('next').NextConfig} */
const nextConfig = {
  // Keep the demo iframe clean; the Next.js issue badge is not part of the product UI.
  devIndicators: false,
  // Same-origin /api on :3100, forwarded to the already-loaded local backend.
  async rewrites() {
    const backendUrl = (
      process.env.MATRIX_BACKEND_URL ||
      "http://127.0.0.1:8100"
    ).replace(/\/$/, "");
    const graphUrl = (process.env.GRAPH_BACKEND_URL || "http://127.0.0.1:4173").replace(/\/$/, "");
    return [
      ...["research-interests", "research-topics", "saved-people", "saved-people/:authorId"].map(path => ({source:`/api/${path}`,destination:`${graphUrl}/api/${path}`})),
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
