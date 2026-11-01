# GitHub Codespace — WebRTC Seminar 04

**Course:** Multimediaprogrammierung — Chapter 04: WebRTC  
**Author:** Renato Fera P. — TU Ilmenau AMS Group  
**Repository:** https://github.com/TUIlmenauAMS/multimedia_programming_seminars

---

## What is this Codespace for?

Demos 1–4 in Chapter 04 are fully self-contained HTML files — open them in
a browser and they work immediately, with no server required.

**Demos 5, 6, and 7 require two servers:**

| Port | Server | Protocol | Used by |
|------|--------|----------|---------|
| 8765 | WebSocket signaling server | WSS (WebSocket over TLS) | Demos 5, 6, 7 |
| 5349 | coturn TURN server | TURNS (TURN over TLS/TCP) | Demos 6, 7 |

This Codespace provisions both servers automatically when it starts.
You do not need to install anything on your own machine.

---

## How to launch the Codespace

### Step 1 — Open the repository on GitHub

Navigate to:
```
https://github.com/TUIlmenauAMS/multimedia_programming_seminars
```

### Step 2 — Create a Codespace

Click the green **Code** button → **Codespaces** tab → **Create codespace on main**.

The Codespace builds a cloud-hosted Linux VM and opens VS Code in your browser.

**First-time build takes about 2 minutes.** It installs:
- coturn (the TURN server)
- Python 3.12
- The `websockets` Python library

On subsequent starts (resume), the build is skipped and startup is instant.

### Step 3 — Wait for the servers to start

The terminal panel at the bottom of VS Code shows the startup log.
You will see:

```
coturn started (TURNS/TCP on port 5349) — PID xxxxx
Signaling server started (WSS on port 8765) — PID xxxxx
════════════════════════════════════════════════════════════
  WebRTC Seminar 04 — Servers Running
════════════════════════════════════════════════════════════
  ...
```

### Step 4 — Find the forwarded port URLs

Click the **Ports** tab in the terminal panel (next to Terminal, Output, Problems).

You will see two forwarded ports:

| Port | Label | Your URL |
|------|-------|----------|
| 8765 | Signaling Server (WSS) | `https://yourname-random-8765.app.github.dev` |
| 5349 | TURN Server (TURNS/TCP) | `https://yourname-random-5349.app.github.dev` |

Hover over each entry and click the **globe icon** (🌐) to copy the public URL.

> **Important:** Every Codespace gets a unique URL. Do not share screenshots of
> your URL — it will not work for someone else.

### Step 5 — Configure the demo pages

**For the signaling server URL (demos 5, 6, 7):**

Take the URL for port 8765 and **replace `https://` with `wss://`**:

```
https://yourname-random-8765.app.github.dev
        ↓ becomes ↓
wss://yourname-random-8765.app.github.dev
```

Paste this WSS URL into the **Codespace WSS URL** field in the demo page.

**For the TURN server (demos 6, 7):**

Take the hostname from the port 5349 URL (everything after `https://`):

```
yourname-random-5349.app.github.dev
```

The demo page builds the TURNS URL automatically from this hostname:

```javascript
urls: 'turns:yourname-random-5349.app.github.dev:5349?transport=tcp',
username: 'webrtc',
credential: 'seminar04'
```

### Step 6 — Open a demo in two browser tabs

Open the same demo HTML file in two browser tabs (or two different browsers).

Both tabs must use the **same room ID** (default: `room1`).

1. **Tab 1** connects first → it becomes the **Caller**
2. **Tab 2** connects second → it becomes the **Callee**

The call is established automatically.

### Step 7 — Stop the Codespace when done

> ⚠️ **GitHub Free accounts include 60 core-hours per month.**
> A 2-core Codespace uses 2 core-hours per clock hour.
> Always stop your Codespace when you are done to avoid wasting your allowance.

To stop:

1. Go to **Code → Codespaces** on the repository page
2. Click the **three-dot menu (⋯)** next to your Codespace
3. Click **Stop codespace**

Or from inside VS Code: press `F1` → type `Codespaces: Stop Current Codespace`.

---

## Infrastructure files

| File | Purpose |
|------|---------|
| `.devcontainer/devcontainer.json` | Defines the VM image, packages, port forwarding, and startup command |
| `codespace/signaling_server.py` | Python WebSocket signaling server (asyncio + websockets) |
| `codespace/coturn.conf` | coturn TURN server configuration |
| `codespace/start.sh` | Generates TLS cert, starts both servers, prints usage instructions |
| `codespace/README.md` | This file |

---

## Troubleshooting

### "Connection refused" on the WSS URL

The signaling server may still be starting. Wait 10–15 seconds and try again.
Check the terminal for the "Signaling server started" message.

### "ICE failed" / no video in Demo 5

Check that both tabs are using the **same WSS URL** and the **same room ID**.
Refresh both tabs and try again.

### No relay candidate in Demo 6

Verify that the TURNS URL is correct (hostname from port 5349, not 8765).
Make sure you are using `?transport=tcp` at the end of the TURNS URL.
Check the coturn log: `tail -f /var/log/coturn/turnserver.log`

### Codespace ports not showing in the Ports tab

Click the **Ports** tab explicitly (it may be hidden behind the Terminal tab).
If ports 8765 or 5349 are not listed, check the terminal for errors from `start.sh`.

### `getUserMedia` permission denied

Demo pages must be served from an HTTPS context. Opening the HTML file
directly from the filesystem (`file://`) will not work for demos that need
`getUserMedia`. Use GitHub Pages
(`https://tuiilmenauams.github.io/multimedia_programming_seminars/`) instead.

---

## TURN credentials

The TURN server is configured with long-term credentials for demo use:

| Field | Value |
|-------|-------|
| Username | `webrtc` |
| Password | `seminar04` |

> **Note:** These credentials are intentionally public — this is a course demo,
> not a production deployment. In production, use time-limited HMAC-SHA1
> credentials that expire automatically. See `coturn.conf` for details.

---

## Why TURNS over TCP?

GitHub Codespaces port forwarding is HTTP-based. It can proxy TCP/TLS traffic
(HTTPS/WSS/TURNS) but **cannot proxy raw UDP datagrams**.

Standard TURN uses UDP (port 3478). TURNS is TURN wrapped in TLS over TCP
(port 5349). Because TURNS is TCP/TLS, it passes through the Codespace
port-forwarding proxy — raw TURN/UDP does not.

This is not a workaround: TURNS over TCP is a valid production configuration,
widely used in enterprise environments that block all UDP traffic. The WebRTC
concepts are identical; only the transport differs.
