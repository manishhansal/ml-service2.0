#!/usr/bin/env python3
"""Extract postgres password from worker DATABASE_URL and fix data-service .env"""
import subprocess
from pathlib import Path

# Get DATABASE_URL from worker
env_output = subprocess.check_output(['docker', 'exec', 'data-service-worker', 'env']).decode()
db_url = None
for line in env_output.splitlines():
    if line.startswith('DATABASE_URL='):
        db_url = line.split('=', 1)[1]
        break

if not db_url:
    print("ERROR: No DATABASE_URL in worker")
    exit(1)

print(f"Worker DB URL found: postgresql+asyncpg://...")

# Extract password
# Format: postgresql+asyncpg://mds_user:PASSWORD@postgres:5432/mds
auth_part = db_url.split('://')[1].split('@')[0]  # mds_user:PASSWORD
pg_pass = auth_part.split(':', 1)[1] if ':' in auth_part else ''

if not pg_pass:
    print("ERROR: Could not extract password")
    exit(1)

print(f"Got postgres password (length={len(pg_pass)})")

# Update .env
env_path = Path('/Users/manishkumar/Desktop/data-service2.0/.env')
content = env_path.read_text()
lines = [l for l in content.splitlines() if not l.startswith('POSTGRES_PASSWORD=')]
lines.append(f'POSTGRES_PASSWORD={pg_pass}')
env_path.write_text('\n'.join(lines) + '\n')
print(f"Updated .env with POSTGRES_PASSWORD")

# Verify the postgres works with this password
try:
    result = subprocess.run(
        ['docker', 'exec', 'data-service-postgres',
         'psql', f'-U', 'mds_user', '-d', 'mds', '-c', 'SELECT 1'],
        capture_output=True, text=True
    )
    print("Postgres connection test:", "OK" if result.returncode == 0 else result.stderr[:80])
except Exception as e:
    print(f"Test failed: {e}")
