#!/bin/bash
# start.sh — starts coturn and the WebSocket signaling server inside the Codespace.
#
# Course: Multimediaprogrammierung — Chapter 04: WebRTC
# Author: Renato Fera P. — TU Ilmenau AMS Group
#
# This script is called automatically by the Codespace on every start via the
# postStartCommand in devcontainer.json:
#   "postStartCommand": "bash codespace/start.sh"
#
# It does three things:
#   1. Generates a self-signed TLS certificate (once, on first start)
#   2. Starts coturn in the background (TURNS on port 5349)
#   3. Starts the Python signaling server in the background (WSS on port 8765)
#
# Both servers run as background processes. The script ends with `wait` to
# keep the postStartCommand alive — if this script exits, the Codespace
# considers the startup complete and stops monitoring the background processes.

set -e  # Exit immediately if any command fails (before the background launches).

# ── TLS certificate paths ──────────────────────────────────────────────────────
# Both coturn and signaling_server.py share the same self-signed certificate.
# The /etc/coturn/ directory is the conventional location for coturn's TLS files.
CERT=/etc/coturn/turn_server_cert.pem
KEY=/etc/coturn/turn_server_pkey.pem

# ── Generate self-signed TLS certificate (first run only) ────────────────────
# The certificate is only generated if it does not already exist.
# On a Codespace resume (not first create), the /etc directory may be reset
# depending on the devcontainer image — the check handles both cases.
if [ ! -f "$CERT" ]; then
    echo "Generating self-signed TLS certificate..."

    # Create the directory if it does not exist.
    mkdir -p /etc/coturn

    # openssl req -x509: generate a self-signed X.509 certificate
    # -newkey rsa:2048:  generate a new 2048-bit RSA key pair
    # -keyout "$KEY":    write the private key to this path
    # -out "$CERT":      write the certificate to this path
    # -days 365:         certificate is valid for 1 year
    # -nodes:            do not encrypt the private key (no passphrase)
    # -subj "/CN=...":   set the Common Name without interactive prompts
    openssl req -x509 -newkey rsa:2048 \
        -keyout "$KEY" -out "$CERT" \
        -days 365 -nodes \
        -subj "/CN=codespace.local"

    echo "Certificate generated at $CERT"
fi

# Ensure the coturn log directory exists.
mkdir -p /var/log/coturn

# ── Resolve the repo root ─────────────────────────────────────────────────────
# The Codespace mounts the repository at /workspaces/<repo-name>.
# We resolve the script's directory to find the repo root regardless of
# the exact repo name.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Start coturn ──────────────────────────────────────────────────────────────
# -c: path to the configuration file
# &:  run in the background (do not block this script)
#
# coturn will log to /var/log/coturn/turnserver.log (as configured in coturn.conf).
# View live logs with: tail -f /var/log/coturn/turnserver.log
turnserver -c "$REPO_ROOT/codespace/coturn.conf" &
COTURN_PID=$!
echo "coturn started (TURNS/TCP on port 5349) — PID $COTURN_PID"

# ── Start the WebSocket signaling server ──────────────────────────────────────
# The signaling server logs to stdout, which appears in the Codespace terminal.
python "$REPO_ROOT/codespace/signaling_server.py" &
SIG_PID=$!
echo "Signaling server started (WSS on port 8765) — PID $SIG_PID"

# ── Print usage instructions ──────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  WebRTC Seminar 04 — Servers Running"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  1. Open the Ports tab in the terminal panel below."
echo "  2. Find the forwarded URLs for port 8765 and port 5349."
echo ""
echo "  For the SIGNALING server (port 8765):"
echo "    Copy the URL and replace https:// with wss://"
echo "    Example: wss://yourname-random-8765.app.github.dev"
echo ""
echo "  For the TURN server (port 5349):"
echo "    Use the URL as-is (https:// host), but paste it into"
echo "    the TURNS host field in the demo page."
echo "    The RTCConfiguration format is:"
echo "      urls: 'turns:<HOST>:5349?transport=tcp'"
echo "      username: 'webrtc'"
echo "      credential: 'seminar04'"
echo ""
echo "  Open Demo 5, 6, or 7 in two browser tabs to start a call."
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Keep this script alive ────────────────────────────────────────────────────
# `wait` blocks until ALL background child processes exit.
# This is essential: if start.sh exits, the Codespace postStartCommand
# completes and the background processes may be orphaned or reaped.
# As long as `wait` is running, the servers stay up.
wait
