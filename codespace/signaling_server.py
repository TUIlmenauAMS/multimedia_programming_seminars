"""Minimal WebSocket signaling server for WebRTC demos.

Course: Multimediaprogrammierung — Chapter 04: WebRTC
Author: Renato Fera P. — TU Ilmenau AMS Group

This server is intentionally minimal: it routes JSON messages between two
peers in the same room. It has no knowledge of SDP or ICE — it is a dumb
router. All the intelligence is in the browser-side WebRTC code.

Message format (JSON):
  { "type": "join",       "room": "<room_id>" }
  { "type": "offer",      "room": "<room_id>", "data": <SDP object> }
  { "type": "answer",     "room": "<room_id>", "data": <SDP object> }
  { "type": "ice",        "room": "<room_id>", "data": <ICE candidate object> }

Server → client messages:
  { "type": "joined" }       — sent to the first peer when it joins
  { "type": "peer-joined" }  — sent to the first peer when the second joins

Run with: python codespace/signaling_server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from pathlib import Path

import websockets
from websockets.asyncio.server import ServerConnection

# ── Logging ──────────────────────────────────────────────────────────────────
# Structured logging to stdout. The Codespace terminal shows these messages
# in real time so you can watch the signaling exchange as it happens.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ── Room registry ─────────────────────────────────────────────────────────────
# rooms maps a room_id (str) to a list of at most 2 active WebSocket connections.
# dict[str, list[ServerConnection]]:
#   key   — room identifier string (e.g. "room1", "lecture")
#   value — list of 1 or 2 WebSocket connections currently in that room
#
# Rooms are created on first join and deleted when the last peer disconnects.
# This server supports only two peers per room (one caller, one callee).
# A production signaling server would add authentication, room capacity limits,
# and persistent room state.
rooms: dict[str, list[ServerConnection]] = {}


# ── Connection handler ────────────────────────────────────────────────────────
async def handler(websocket: ServerConnection) -> None:
    """Handle one WebSocket connection for its entire lifetime.

    This coroutine is called once per incoming connection. It loops over all
    messages from that client, routes them to the other peer in the same room,
    and cleans up when the connection closes (normally or on error).

    Args:
        websocket: The WebSocket connection object for this client.
    """
    # Track which room this peer belongs to so we can clean up on disconnect.
    room_id: str | None = None

    try:
        # Iterate over all messages from this WebSocket.
        # This loop runs until the connection closes.
        async for raw in websocket:
            # Parse the JSON message. All messages from the client must be
            # valid JSON — if parsing fails, the exception propagates and
            # the connection is closed with an error.
            message: dict = json.loads(raw)
            msg_type: str = message.get('type', '')
            room_id = message.get('room', 'default')

            if msg_type == 'join':
                # ── JOIN: Add this peer to the room ──────────────────────────
                # Create the room entry if it does not exist yet.
                if room_id not in rooms:
                    rooms[room_id] = []

                # Reject a third peer — this server supports exactly 2 per room.
                if len(rooms[room_id]) >= 2:
                    await websocket.send(json.dumps({'type': 'error', 'message': 'Room is full'}))
                    log.warning(f"Room '{room_id}' is full — rejecting third peer")
                    continue

                rooms[room_id].append(websocket)
                peer_count = len(rooms[room_id])
                log.info(f"Peer joined room '{room_id}' ({peer_count}/2 peers)")

                if peer_count == 1:
                    # First peer — acknowledge the join and wait for a second peer.
                    await websocket.send(json.dumps({'type': 'joined'}))

                elif peer_count == 2:
                    # Second peer has arrived — tell the FIRST peer (the caller)
                    # so it can start the WebRTC offer/answer exchange.
                    # The second peer (callee) will receive the offer via the
                    # normal message routing below.
                    first_peer = rooms[room_id][0]
                    await first_peer.send(json.dumps({'type': 'peer-joined'}))
                    log.info(f"Room '{room_id}' is now full — notified first peer")

            else:
                # ── ROUTE: Forward the message to the OTHER peer ──────────────
                # For offer, answer, and ice messages, find every peer in the
                # room that is NOT the sender and forward the raw JSON string.
                # We forward the raw string (not re-serialised) to avoid any
                # accidental modification of the SDP or ICE candidate data.
                peers = rooms.get(room_id, [])
                forwarded = False
                for peer in peers:
                    if peer is not websocket:
                        await peer.send(raw)
                        forwarded = True
                        log.info(f"Routed '{msg_type}' in room '{room_id}'")

                if not forwarded:
                    # No other peer in the room yet — the message cannot be delivered.
                    log.warning(f"No peer to route '{msg_type}' in room '{room_id}'")

    except websockets.exceptions.ConnectionClosedOK:
        # Normal disconnect (client closed the connection cleanly).
        # No error logging needed — this is expected behaviour.
        pass
    except websockets.exceptions.ConnectionClosedError as e:
        log.warning(f"Connection closed with error in room '{room_id}': {e}")
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON from client in room '{room_id}': {e}")
    except Exception as e:
        log.error(f"Unexpected handler error in room '{room_id}': {e}")
    finally:
        # ── CLEANUP: Remove this peer from its room on disconnect ─────────────
        # This runs whether the connection closed normally, with an error,
        # or due to an exception — the finally block always executes.
        if room_id and room_id in rooms:
            rooms[room_id] = [p for p in rooms[room_id] if p is not websocket]
            remaining = len(rooms[room_id])
            log.info(f"Peer left room '{room_id}' ({remaining} remaining)")
            # Delete the room entry when it is empty to keep the dict clean.
            if not rooms[room_id]:
                del rooms[room_id]
                log.info(f"Room '{room_id}' deleted (empty)")


# ── TLS context ───────────────────────────────────────────────────────────────
def make_ssl_context() -> ssl.SSLContext:
    """Create a TLS server context using the self-signed certificate.

    The certificate is generated by start.sh using openssl req -x509.
    Both coturn and this signaling server share the same certificate/key pair,
    stored at /etc/coturn/turn_server_{cert,pkey}.pem.

    In production, replace these with a certificate from Let's Encrypt or
    your organisation's CA. The paths are the only thing that changes.

    Returns:
        An ssl.SSLContext configured for TLS server use.

    Raises:
        FileNotFoundError: If the certificate or key file does not exist.
            Run start.sh first to generate the certificate.
    """
    cert = Path('/etc/coturn/turn_server_cert.pem')
    key  = Path('/etc/coturn/turn_server_pkey.pem')

    if not cert.exists() or not key.exists():
        raise FileNotFoundError(
            "TLS certificate not found. Run 'bash codespace/start.sh' first "
            "to generate the self-signed certificate."
        )

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
    return ctx


# ── Entry point ───────────────────────────────────────────────────────────────
async def main() -> None:
    """Start the WSS signaling server and run until interrupted."""
    ssl_context = make_ssl_context()
    log.info("Signaling server starting on wss://0.0.0.0:8765")
    log.info("Waiting for connections...")

    # websockets.serve() creates an async context manager that runs the server.
    # `handler` is called once for each new WebSocket connection.
    # Listening on 0.0.0.0 makes the server reachable on all network interfaces,
    # including the Codespace's forwarded port interface.
    async with websockets.serve(handler, '0.0.0.0', 8765, ssl=ssl_context):
        # create_future() returns a Future that never completes — this keeps
        # the server running until the process receives a signal (Ctrl-C, SIGTERM).
        await asyncio.get_running_loop().create_future()


if __name__ == '__main__':
    asyncio.run(main())
