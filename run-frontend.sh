#!/usr/bin/env bash
# Run the Next.js frontend locally
set -e
cd "$(dirname "$0")/frontend"
npm run dev -- -p 3000
