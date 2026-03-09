#!/usr/bin/env bash
# Run Next.js frontend the same way as systemd.
set -e
cd "$(dirname "$0")/frontend"

# Ensure dependencies are present
if [ ! -d "node_modules" ]; then
  npm ci
fi

# Ensure production build exists
if [ ! -d ".next" ]; then
  npm run build
fi

echo "Starting frontend on port 3000 ..."
exec npm run start -- -p 3000
