# Phase 7 — Dashboard

Read `CLAUDE.md` before starting. Phase 6 must pass.

## Goal

A browser page on localhost that shows the prediction live when I load into a
match. This is what gets screenshotted for the portfolio, so it should look
deliberate — but it is a readout, not a design project.

## Build

**1. Server — `valwr/web/app.py`**

FastAPI. **Binds to `127.0.0.1` only** — never `0.0.0.0`. This serves other
players' data and must not be reachable from the network. See
`docs/ETHICS-AND-TOS.md`.

- Websocket endpoint pushing predictions as they are produced
- Wraps the Phase 6 watcher; one process, `python -m valwr.web`
- Serves the static front end

**2. Front end — `valwr/web/static/`**

Vanilla HTML/CSS/JS, no build step, no framework. Keep it that way.

Layout, roughly:

```
+---------------------------------------------------------------+
|  ASCENT                                    agent select · 0:23 |
+---------------------------------------------------------------+
|                                                               |
|     YOUR TEAM  58.3%   ############------  41.7%  ENEMY       |
|                                                               |
+---------------------------------------------------------------+
|  WHY                                                          |
|   +6.1  Jett on Ascent — 71% over 24 games                    |
|   +3.2  team rating spread (one strong outlier)               |
|   -2.4  no controller in comp                                 |
|   -1.8  enemy 3-stack                                         |
+---------------------------------------------------------------+
|  YOUR TEAM                    |  ENEMY                        |
|  Jett     Imm1  rating 1.24   |  Omen     Imm2  rating 1.31   |
|  Omen     Imm2  rating 1.18   |  Sova     Imm1  rating 1.09   |
|  ...                          |  Raze     ?     unresolved    |
+---------------------------------------------------------------+
|  confidence: moderate — 9/10 players resolved                 |
+---------------------------------------------------------------+
```

Requirements:

- Both team probabilities, prominent and immediately readable
- **Top positive and negative factors in plain language**, translated from SHAP.
  "Jett on Ascent — 71% over 24 games" is useful; `map_agent_wr_shrunk: +0.061`
  is not. This translation layer is the difference between a demo and a tool.
- Per-player cards: agent, rank, rating, relevant map/agent history
- **Unresolved players visibly marked**, and a confidence indicator reflecting
  how much data the prediction actually had. Hiding uncertainty behind a
  confident-looking number is the one thing this UI must not do.
- Updates live on match start with no refresh
- Readable in a dark room — this is used mid-game. Dark by default.

**3. States**

Handle and show: client not running, not in a match, agent select (partial
locks), in game, resolving players, prediction ready, degraded. Each should look
intentional rather than blank.

## Constraints

- `127.0.0.1` only.
- No build step, no npm, no framework.
- No in-game overlay — separate browser window. See `docs/ETHICS-AND-TOS.md`.
- Do not reimplement prediction logic in the front end. The server sends a
  finished payload; the page renders it.

## Acceptance criteria

`python -m valwr.web`, open `localhost:8000`, load into a match, and the page
updates live with a prediction, factors, and player cards — no refresh. The
degraded path renders sensibly with unresolved players.

Take a screenshot for the README.

## Do not build yet

No LLM coaching — that is Phase 8. Factors here come from SHAP, translated by
template.
