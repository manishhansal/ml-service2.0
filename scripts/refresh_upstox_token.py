#!/usr/bin/env python3
"""
scripts/refresh_upstox_token.py
────────────────────────────────
Automates the daily Upstox OAuth token refresh.

Steps:
  1. Reads UPSTOX_API_KEY + UPSTOX_API_SECRET from data-service2.0/.env
  2. Prints the Upstox login URL
  3. Starts a local HTTP server on port 8200 path /v1/auth/upstox/callback
     (or an alternate port 8201 if 8200 is in use by data-service)
  4. Waits for the browser to redirect with the auth code
  5. Exchanges the auth code for a new access token
  6. Updates UPSTOX_ACCESS_TOKEN in BOTH:
       - /Users/manishkumar/Desktop/data-service2.0/.env
       - /Users/manishkumar/Desktop/ml-service2.0/.env
  7. Restarts data-service2.0 containers to pick up the new token

Usage:
    python3 scripts/refresh_upstox_token.py

The script opens the login URL automatically in your default browser.
After logging into Upstox, the auth code is captured automatically.

Token validity: Upstox access tokens are valid for ~24 hours.
Run this script each morning before market open (before 09:00 IST).
"""
from __future__ import annotations

import json
import os
import re
import sys
import subprocess
import threading
import webbrowser
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_SVC_DIR = Path("/Users/manishkumar/Desktop/data-service2.0")
ML_SVC_DIR   = Path("/Users/manishkumar/Desktop/ml-service2.0")
DATA_ENV     = DATA_SVC_DIR / ".env"
ML_ENV       = ML_SVC_DIR / ".env"

# Upstox OAuth endpoints
UPSTOX_TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
REDIRECT_PORT    = 8201   # use a temp port so data-service can stay running
REDIRECT_URI     = f"http://localhost:{REDIRECT_PORT}/v1/auth/upstox/callback"


# ── .env helpers ──────────────────────────────────────────────────────────────
def read_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def update_env_key(path: Path, key: str, value: str) -> None:
    """Update or insert KEY=VALUE in a .env file, preserving all comments."""
    text = path.read_text() if path.exists() else ""
    pattern = rf"^{re.escape(key)}=.*$"
    replacement = f"{key}={value}"
    if re.search(pattern, text, re.MULTILINE):
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    else:
        text = text.rstrip("\n") + f"\n{replacement}\n"
    path.write_text(text)
    print(f"  ✓ Updated {key} in {path.name}")


# ── Local callback server ─────────────────────────────────────────────────────
_auth_code: list[str] = []   # mutable shared state
_server_ready = threading.Event()


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        code   = params.get("code")
        error  = params.get("error")

        if code:
            _auth_code.append(code)
            body = b"<h2>Upstox auth code received! You can close this tab.</h2>"
            self.send_response(200)
        else:
            body = f"<h2>Error: {error}</h2>".encode()
            self.send_response(400)

        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # silence default access log
        pass


def _run_server(server: HTTPServer):
    _server_ready.set()
    while not _auth_code:
        server.handle_request()


# ── Token exchange ────────────────────────────────────────────────────────────
def exchange_code(code: str, api_key: str, api_secret: str) -> str | None:
    data = urllib.parse.urlencode({
        "code":          code,
        "client_id":     api_key,
        "client_secret": api_secret,
        "redirect_uri":  REDIRECT_URI,
        "grant_type":    "authorization_code",
    }).encode()
    req = urllib.request.Request(
        UPSTOX_TOKEN_URL,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept":       "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read())
            return payload.get("access_token")
    except Exception as exc:
        print(f"  ERROR exchanging auth code: {exc}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print("  UPSTOX TOKEN REFRESH")
    print("=" * 65)

    # Read credentials from data-service .env
    env = read_env(DATA_ENV)
    api_key    = env.get("UPSTOX_API_KEY")
    api_secret = env.get("UPSTOX_API_SECRET")
    old_token  = env.get("UPSTOX_ACCESS_TOKEN", "")

    if not api_key or not api_secret:
        print("ERROR: UPSTOX_API_KEY or UPSTOX_API_SECRET not found in", DATA_ENV)
        sys.exit(1)

    # Show current token expiry
    if old_token:
        try:
            import base64
            pad = old_token.split(".")[1] + "=="
            payload = json.loads(base64.b64decode(pad))
            exp_dt  = datetime.fromtimestamp(payload["exp"])
            hrs     = (exp_dt - datetime.now()).total_seconds() / 3600
            print(f"  Current token expires: {exp_dt.strftime('%Y-%m-%d %H:%M IST')} "
                  f"({'EXPIRED' if hrs < 0 else f'{hrs:.1f}h remaining'})")
        except Exception:
            pass

    # Start callback server
    server = HTTPServer(("localhost", REDIRECT_PORT), _CallbackHandler)
    t = threading.Thread(target=_run_server, args=(server,), daemon=True)
    t.start()
    _server_ready.wait(timeout=3)

    # Build login URL (uses temp redirect port so data-service keeps running)
    login_params = {
        "client_id":     api_key,
        "redirect_uri":  REDIRECT_URI,
        "response_type": "code",
        "state":         "ml-service-refresh",
    }
    login_url = ("https://api.upstox.com/v2/login/authorization/dialog?"
                 + urllib.parse.urlencode(login_params))

    print(f"\n  Opening Upstox login in your browser...")
    print(f"  Login URL:\n  {login_url}\n")
    webbrowser.open(login_url)
    print("  Waiting for callback (log in → allow → this window captures the code)...")

    # Wait up to 120 seconds
    t.join(timeout=120)
    server.server_close()

    if not _auth_code:
        print("\n  TIMEOUT: No auth code received in 120 seconds.")
        print("  To retry: run this script again, or paste the callback URL below:")
        manual = input("  Paste callback URL (or press Enter to abort): ").strip()
        if manual:
            params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(manual).query))
            code   = params.get("code")
            if code:
                _auth_code.append(code)
        if not _auth_code:
            print("  Aborted.")
            sys.exit(1)

    code = _auth_code[0]
    print(f"\n  Auth code received ✓ (length={len(code)})")

    # Exchange for access token
    print("  Exchanging auth code for access token...")
    new_token = exchange_code(code, api_key, api_secret)
    if not new_token:
        print("  ERROR: Token exchange failed. Check UPSTOX_API_SECRET.")
        sys.exit(1)

    # Decode expiry
    try:
        import base64
        pad     = new_token.split(".")[1] + "=="
        payload = json.loads(base64.b64decode(pad))
        exp_dt  = datetime.fromtimestamp(payload["exp"])
        print(f"  New token expires: {exp_dt.strftime('%Y-%m-%d %H:%M IST')}")
    except Exception:
        print("  New token received (could not decode expiry)")

    # Update .env files
    print("\n  Updating .env files...")
    update_env_key(DATA_ENV, "UPSTOX_ACCESS_TOKEN", new_token)
    update_env_key(ML_ENV,   "UPSTOX_ACCESS_TOKEN", new_token)

    # Restart data-service containers
    print("\n  Restarting data-service2.0 to pick up new token...")
    try:
        result = subprocess.run(
            ["docker", "compose", "restart", "api", "worker", "scheduler"],
            cwd=str(DATA_SVC_DIR),
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print("  data-service2.0 restarted ✓")
        else:
            print(f"  Warning: restart returned non-zero: {result.stderr[:200]}")
    except Exception as exc:
        print(f"  Warning: could not restart containers automatically: {exc}")
        print("  Run manually: cd ~/Desktop/data-service2.0 && docker compose restart api worker scheduler")

    print("\n" + "=" * 65)
    print("  Token refresh complete ✓")
    print("  Run this script again tomorrow before market open (09:00 IST)")
    print("=" * 65)


if __name__ == "__main__":
    main()
