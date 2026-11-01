#!/bin/bash
# start.sh — starts coturn and the WebSocket signaling server inside the Codespace.
#
# Course: Multimediaprogrammierung — Chapter 04: WebRTC
# Author: Renato Fera P. — TU Ilmenau AMS Group
#
# Called automatically via postStartCommand in devcontainer.json.
# Runs as the 'vscode' user. /etc/coturn and /var/log/coturn are
# made world-writable by onCreateCommand (which runs as root) so
# this script can write there without sudo.

set -e

# ── TLS certificate paths ─────────────────────────────────────────────────────
# Stored under /etc/coturn/ (made writable by onCreateCommand).
# Both coturn and signaling_server.py share this certificate.
CERT=/etc/coturn/turn_server_cert.pem
KEY=/etc/coturn/turn_server_pkey.pem

# ── Generate self-signed TLS certificate (once per Codespace lifetime) ────────
if [ ! -f "$CERT" ]; then
  echo "Generating self-signed TLS certificate..."
  openssl req -x509 -newkey rsa:2048 \
    -keyout "$KEY" -out "$CERT" \
    -days 365 -nodes \
    -subj "/CN=codespace.local"
  echo "Certificate generated at $CERT"
else
  echo "TLS certificate already exists — skipping generation."
fi

# ── Resolve the repo root ─────────────────────────────────────────────────────
# The Codespace mounts the repo at /workspaces/<repo-name>.
# BASH_SOURCE[0] is this script; its parent is the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Repo root: $REPO_ROOT"

# ── Start coturn ──────────────────────────────────────────────────────────────
turnserver -c "$REPO_ROOT/codespace/coturn.conf" &
COTURN_PID=$!
echo "coturn started (TURNS/TCP on port 5349) — PID $COTURN_PID"

# ── Start the WebSocket signaling server ──────────────────────────────────────
python "$REPO_ROOT/codespace/signaling_server.py" &
SIG_PID=$!
echo "Signaling server started (WSS on port 8765) — PID $SIG_PID"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  WebRTC Seminar 04 — Servers Running"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  1. Open the Ports tab in the terminal panel."
echo "  2. Find port 8765 — copy its URL and replace https:// with wss://"
echo "     Paste that wss:// URL into the demo page."
echo ""
echo "  Open Demo 5 in two browser tabs and use the same Room ID."
echo "════════════════════════════════════════════════════════════"
echo ""

# Keep this script alive so postStartCommand does not exit.
# If this script exits, background processes may be orphaned.
wait
