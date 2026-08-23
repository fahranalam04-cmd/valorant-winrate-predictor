# Phase 6 — Live client integration

Read `CLAUDE.md`, `docs/API-NOTES.md`, and **`docs/ETHICS-AND-TOS.md`** before
starting. Phase 5 must pass — there needs to be a trained model to serve.

## Goal

Detect that I have loaded into a match, resolve the ten players, and produce a
prediction before the first round.

## Ban safety — read this first

`docs/ETHICS-AND-TOS.md` is a ban-safety document, not boilerplate. The rule is
**observe, never act**:

- Read-only. Never POST to the client API.
- **Never automate agent selection** — instalockers are the most common cause of
  bans in this category of tool. Do not select, lock, hover, or dodge.
- No process memory reading, no injection, no DLLs, no hooks.
- No in-game overlay. The dashboard is a separate browser window.

If any part of this phase seems to require writing to the client, stop and ask.
It does not.

## Build

**1. Lockfile auth — `valwr/live/lockfile.py`**

Parse `%LOCALAPPDATA%\Riot Games\Riot Client\Config\lockfile` —
`name:pid:port:password:protocol`.

HTTP Basic against `https://127.0.0.1:{port}`, username literally `riot`.
The cert is self-signed, so TLS verification is disabled **for that localhost
connection only** — never globally, and never on the HenrikDev client. Make that
explicit in the code so it cannot be copied somewhere it does not belong.

Handle the client not running: clear message, retry loop, no crash.

**2. Identity and region — `valwr/live/session.py`**

- `GET /entitlements/v1/token` — access token + entitlements token
- `GET /chat/v1/session` — my own PUUID
- Region/shard from the `-ares-deployment=` argument on the running client
  process, read via `psutil`. Do not hardcode it — the account may move region.

**3. Match detection — `valwr/live/watch.py`**

Subscribe to the local websocket, event
`OnJsonApiEvent_riot-messaging-service_v1_message`. Watch URI prefixes:

| Prefix | Phase |
|---|---|
| `ares-pregame/pregame/v1/matches/` | agent select |
| `ares-core-game/core-game/v1/matches/` | in game |

Reconnect on drop. Include a polling fallback via `Pregame_GetPlayer` /
`CoreGame_FetchPlayer` — websockets drop, and silently missing a match is worse
than a slightly late prediction.

**4. Roster — `valwr/live/roster.py`**

`Pregame_GetMatch` / `CoreGame_FetchMatch` → ten PUUIDs, locked agents, teams.

Note that in pregame, agents may not all be locked yet. Handle partial agent
information: predict with what is known, update as locks come in.

**5. The rate-limit squeeze — `valwr/live/resolve.py`**

This is the hard part of the phase. Ten unknown PUUIDs, each needing history,
against 30 req/min, during ~30 seconds of agent select. **This cannot be solved
by fetching faster.**

- **Cache first.** History from hours ago is fine. Most lobbies contain players
  already in the database from the crawl — measure the real hit rate and report
  it.
- **Priority order.** My team first, then enemies. Partial output beats none.
- **Degrade, do not fail.** Predict from whoever resolved. Widen the confidence
  band to reflect missing data, and mark which players are unresolved so the
  uncertainty is visible rather than hidden.
- **Pre-warm.** Queue recent teammates and opponents for background refresh
  between matches, when there is spare rate budget.

**6. Prediction — `valwr/live/predict.py`**

Roster → features → model → calibrated probability + SHAP attributions.

**The features must be built by the same code path as training.** If live
feature construction diverges from Phase 4's, the model receives inputs it was
never trained on and the output is garbage that looks plausible. Reuse
`valwr/features/build.py` directly rather than reimplementing for the live case.

Assert the live feature vector has the same shape and column order as training.

**7. CLI**

`python -m valwr.live` — watch, and print predictions to the terminal.
Phase 7 adds the browser front end on top of this.

## Acceptance criteria

Load into a real or custom match. Within agent select:

1. Ten players resolve (or fewer, explicitly marked)
2. A calibrated prediction is produced with top contributing factors
3. The degraded path works — test it by clearing the cache for a few PUUIDs
4. No 429s

## Do not build yet

No web UI — terminal output only. Phase 7 does the dashboard.
