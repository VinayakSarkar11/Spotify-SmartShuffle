#!/bin/bash
set -e

# Ensure runtime directories exist.
# On Railway the persistent volume is mounted at /app/data — mkdir is a no-op
# when the dirs already exist, so this is safe to run on every boot.
cd "$(dirname "$(dirname "$0")")"   # repo root regardless of cwd
mkdir -p data/users logs

cd web
exec uvicorn app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --proxy-headers \
  --forwarded-allow-ips "*" \
  --workers 1
