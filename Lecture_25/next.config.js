/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: "http://localhost:8005/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
