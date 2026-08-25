# API Notes

Every endpoint here was verified against live documentation. **Do not invent
endpoints or fields.** If something you need is not in this file, check the live
docs and add it here with a note, rather than guessing at a path.

---

## HenrikDev API — bulk historical data

Unofficial VALORANT API. This is the only practical source of bulk match
history for a third-party project.

- **Base URL:** `https://api.henrikdev.xyz`
- **Auth:** `Authorization: HDEV-xxxxxxxx` — the raw key, **no `Bearer` prefix**
- **Keys:** generated at <https://api.henrikdev.xyz/dashboard/>, which requires
  joining their Discord and describing your use case
- **OpenAPI spec:** <https://api.henrikdev.xyz/docs>
- **Docs:** <https://docs.henrikdev.xyz>

### Rate limits — the binding constraint on this project

| Tier | Limit | Notes |
|---|---|---|
| Basic | **30 req/min** | educational projects, private bots |
| Enhanced | **90 req/min** | public bots/sites; approval required, takes days |
| Production | negotiated | requires Patreon support, rarely granted |

Exceeding returns **`429`** with body `Rate Limited`. Honour `Retry-After`.

**The stated ceiling is not the request budget.** Every response carries the
server's own accounting:

```
x-ratelimit-limit: 30      ratelimit-policy: "per1min";q=30;w=60
x-ratelimit-remaining: 29  x-ratelimit-bucket: <uuid>
x-ratelimit-reset: 60
```

Measured live: **15 matchlist requests consumed ~26 quota units** — a full
`size=10` matchlist costs roughly **2 units**, not 1, because it fans out to
Riot rather than serving cache. HenrikDev's own docs foreshadow this ("some
calls require fetches from Riot while others use cache or local data").

So do not model the limit client-side. `valwr/collect/limiter.py` reconciles
against `x-ratelimit-remaining` after every response and only ever revises
*down*. A client-side 30/min token bucket earned three 429s while averaging
under 3 req/min; the adaptive version has earned none.

Responses also carry `x-cache-status` and `x-cache-ttl: 300` — HenrikDev
caches for five minutes, which is worth knowing for the live path's re-fetches.

### Quota semantics, measured

Probing every 10 seconds settles how the window behaves:

```
t=1   remaining 29  reset 60
t=13  remaining 28  reset 48
t=59  remaining 24  reset  2
t=71  remaining 29  reset 60     <- jumps back to full
```

`reset` **counts down** to the boundary; it is not the window length. The
window is **fixed**, not rolling -- the allowance refills all at once. So
unused quota expires, holding a reserve is waste, and an exhausted window
should sleep exactly `reset` seconds.

### A fresh matchlist costs ~10 units, not 1

This is the number that governs everything. Three different limiter designs
were tried -- a token bucket, adaptive pacing, and a fixed-window limiter --
and all three landed at **2.6-3.2 req/min**:

| Implementation | req/min | implied cost |
|---|---|---|
| Token bucket | 3.2 | 9.4 units |
| Adaptive pacing | 2.8 | 10.7 units |
| Fixed window | 2.6 | 11.5 units |

Three designs converging on the same number is not three failures; it is the
ceiling. At 30 units per minute and ~10 units per fresh matchlist, **~3
req/min is the Basic tier's hard limit.** Cached repeats bill ~1 unit, which is
why a naive probe of the same PUUID looks far cheaper than reality.

The long gaps in a crawl are therefore not waste to optimise away -- they are
the window genuinely spent. **There is no client-side optimisation left.** The
only lever is the Enhanced key: 90 units/min is ~8.6 req/min, roughly 3x.

### Real throughput (basic tier, measured)

| | |
|---|---|
| Sustained requests | **~3.4 req/min** |
| New matches per request | ~7.6 |
| New matches per minute | ~26 |
| Time for 40k matches | **~26 hours** of crawling |

The Enhanced key (90/min stated) should be roughly 3x this. That is the
difference between one overnight run and most of a week, which is why applying
early matters more than any code optimisation.

The maintainer states plainly that this API "is not designed to be used in
production apps" and is not intended for large analytics projects. Treat the
limit as a hard design constraint, not an obstacle: cache everything, crawl
politely, make the collector resumable, and never re-fetch what you already
have. Do not attempt to evade limits with multiple keys or proxies.

Version note: prefer **v4** where it exists. v2 is slated for deprecation. v4 is
a Rust rewrite with a more streamlined data layout.

### Endpoints in use

**Matchlist by PUUID** — the crawler's workhorse.
```
GET /valorant/v4/by-puuid/matches/{region}/{platform}/{puuid}
GET /valorant/v4/matches/{region}/{platform}/{name}/{tag}
```
Query params: `size` (default 10), `start` (pagination, v4.1.0+), `map`,
`mode`, `queue`.

One call returns up to `size` full match objects, each containing all ten
players. That is the efficiency lever — a single request can yield 10 matches ×
10 players of usable data. Filter `mode=competitive` for the training set.

**Single match by ID**
```
GET /valorant/v4/match/{region}/{matchid}
GET /valorant/v2/match/{matchid}          # deprecated, avoid
```

**Leaderboard** — crawler seed.
```
GET /valorant/v3/leaderboard/{region}/{platform}
```
Returns `puuid`, `name`, `tag`, `leaderboard_rank`, `tier`, `rr`, `wins`,
`is_banned`, `is_anonymized`, `updated_at`, plus per-tier thresholds. Can filter
by `puuid` **or** `name`+`tag`, not both. Season filter accepts a short code
(`e1a1`, `e2a3`, …) **or** `season_id`, not both.

Note `is_anonymized` — anonymised leaderboard entries have no usable PUUID.
Skip them rather than letting them poison the frontier queue.

**Stored matches** — lighter payload, useful when you only need outcomes.
```
GET /valorant/v1/by-puuid/stored-matches/{region}/{puuid}
GET /valorant/v1/stored-matches/{region}/{name}/{tag}
```

**MMR history** — rank trajectory over time.
```
GET /valorant/v2/by-puuid/stored-mmr-history/{region}/{platform}/{puuid}
GET /valorant/v1/by-puuid/stored-mmr-history/{region}/{puuid}
```
v2 adds `refunded_rr` and `was_derank_protected`.

**Account lookup** — resolve a Riot ID to a PUUID.
```
GET /valorant/v2/account/{name}/{tag}
GET /valorant/v2/by-puuid/account/{puuid}
```

### Verified v4 match response shape

Confirmed against a real response, not inferred from docs. Top-level keys:

```
metadata, players, teams, rounds, kills, coaches, observers
```

The published docs list only a fraction of this. The `rounds` and `kills`
arrays in particular are far richer than advertised, and they unlock most of
the rating metric.

**`metadata`**
```
match_id  map{id,name}  game_version  game_length_in_ms  started_at (ISO string)
is_completed  queue{id,name,mode_type}  season{id,short}  platform  region
cluster  premier  party_rr_penaltys[]
```
`started_at` is an **ISO 8601 string**, not an epoch int. The `matches` table
stores an INTEGER, so Phase 2 must parse it. Getting this wrong silently
breaks every time-gated feature, so assert on it.

`game_version` gives patch-level granularity, finer than `season`.
`is_completed` is a data-quality flag worth respecting.

**`players[]`** (10 per match)
```
puuid  name  tag  team_id  platform  party_id  account_level
agent{id,name}  tier{id,name}
stats{score,kills,deaths,assists,headshots,bodyshots,legshots,damage{...}}
ability_casts{grenade,ability1,ability2,ultimate}
behavior{afk_rounds, friendly_fire{...}, rounds_in_spawn}
economy{spent{...}, loadout_value{...}}
session_playtime_in_ms
customization{...}
```

**`teams[]`** — `team_id`, `won`, `rounds{won,lost}`

**`rounds[]`** (~26 per match)
```
id  result  ceremony  winning_team  plant  defuse  stats[10]
```
`stats` is **per-player, per-round** -- which is what makes true ADR and a real
KAST possible rather than approximated. `ceremony` carries direct labels for
clutches and aces. `plant`/`defuse` give spike involvement.

**`kills[]`** (~195 per match)
```
time_in_round_in_ms  time_in_match_in_ms  round
killer{puuid,name,tag,team}  victim{...}  assistants[]
weapon{id,name,type}  location{x,y}  player_locations[9]  secondary_fire_mode
```
Kill timings plus killer/victim identity give first bloods, first deaths,
trade participation, and multi-kills directly. `player_locations` records every
other player's position at the moment of each kill -- unused for pre-match
prediction, but it is 36% of the response size.

### Response size

**~450 KB per match uncompressed.** At 50k matches that is ~23 GB of JSON,
which does not fit a laptop comfortably. zlib level 6 takes it to **7.4%
(~1.7 GB)**, so `raw_response.body` is a compressed BLOB -- see
`valwr/store/raw.py`. Compressing rather than dropping fields keeps the
response verbatim, which is the point of the raw layer.

### Values

- `region`: `na`, `eu`, `ap`, `kr`, `latam`, `br` — this project uses `na`
- `platform`: `pc`, `console` — this project uses `pc`

---

## valorant-api.com — static game metadata

Community content API. **No key required.** Pull once into reference tables;
refresh only when a new agent or map ships.

- **Base URL:** `https://valorant-api.com/v1`
- `GET /agents?isPlayableCharacter=true` — agents, UUIDs, **roles**, abilities
- `GET /maps` — maps and UUIDs
- `GET /competitivetiers` — rank tier names and numeric ordering
- `GET /seasons` — episode/act boundaries, for the patch-era feature

The agent→role mapping from `/agents` drives every composition feature
(duelist count, has-controller, role balance, off-role penalty). Do not hardcode
a role table; agents get reworked.

Rank tier ordering from `/competitivetiers` is what makes `tier` numerically
comparable. Do not assume the integer encoding is stable across episodes —
resolve names through this endpoint.

---

## Riot local client API — live match detection

Runs on the machine playing the game. Used only to answer "who are the ten
players in the match I just loaded into". **Read-only** — see
[ETHICS-AND-TOS.md](ETHICS-AND-TOS.md), which is a ban-safety document, not a
formality.

Reference: <https://valapidocs.techchrism.me/>

### Lockfile

```
%LOCALAPPDATA%\Riot Games\Riot Client\Config\lockfile
```
Present only while the client is running. Single line, colon-separated:
```
name:pid:port:password:protocol
```

Authenticate with HTTP Basic, username literally `riot`, password from the
lockfile, against `https://127.0.0.1:{port}`. The certificate is self-signed, so
TLS verification must be disabled **for that localhost connection only** — never
globally, and never for HenrikDev calls.

### Three corrections from a live client

Verified against a running game; each of these costs an hour if taken from the
docs instead.

**The lockfile existing does not mean VALORANT is running.** It is written by
the Riot Client launcher, which is up most of the time. With the launcher
alone, `/help` exposes seven functions (`Exit`, `Help`, `Subscribe`, …) and
every endpoint here returns 404. With the game running it exposes ~1,270.
Check capability, not the file.

**`X-Riot-ClientVersion` is required** on the glz endpoints. Without it they
return an opaque `400` that reads like a bad URL. It comes from
`/product-session/v1/external-sessions` — but that map also holds a `host_app`
entry whose version is the literal string `"0"`, and sending that placeholder
produces the same 400. Skip it and take the real session's 16-character value.

**Take the shard from `/riotclient/region-locale`, not `/chat/v1/session`.**
The chat session reports its XMPP server — `la1` on an NA account — which
builds a glz host that does not exist. There is also no need to scrape
`-ares-deployment` off the process command line as suggested below; the API
reports the region directly.

`404 RESOURCE_NOT_FOUND` from pregame or core-game is the normal "not in a
match" answer, not an error.

### Tokens and identity

```
GET /entitlements/v1/token        # local: access token + entitlements token
GET /chat/v1/session              # local: your own puuid
```
Remote `pd`/`glz` calls need both the access token (`Authorization: Bearer …`)
and the entitlements token (`X-Riot-Entitlements-JWT: …`).

Region/shard comes from the `-ares-deployment=` argument on the running client
process — read it via `psutil` rather than hardcoding, so the app does not break
if the account moves region.

### Match detection

Subscribe to the local websocket and watch:
```
OnJsonApiEvent_riot-messaging-service_v1_message
```
Match the event URI prefix:

| URI prefix | Phase |
|---|---|
| `ares-pregame/pregame/v1/matches/` | agent select |
| `ares-core-game/core-game/v1/matches/` | in game |

Then fetch the roster:
```
GET  {glz}/pregame/v1/matches/{matchid}        # Pregame_GetMatch
GET  {glz}/core-game/v1/matches/{matchid}      # CoreGame_FetchMatch
```
These give the ten PUUIDs, locked agents, and team assignment — everything the
feature builder needs as input.

Polling fallback: `Pregame_GetPlayer` / `CoreGame_FetchPlayer` return the
current match ID on request, for when the websocket connection drops.

### The rate-limit squeeze at match start

Ten unknown PUUIDs, each needing history, against 30 req/min, in the ~30 seconds
of agent select. This cannot be solved by fetching faster. Solve it by:

1. **Cache first.** History from hours ago is fine. Most lobbies contain players
   already in the local database from the crawl.
2. **Priority order.** Fetch your own team first, then enemies — partial output
   beats no output.
3. **Degrade, do not fail.** Predict from whoever resolved, and widen the
   confidence band to reflect missing data. Show which players are unresolved.
4. **Pre-warm.** Queue recent teammates and opponents for background refresh
   between matches, when there is rate budget to spare.

---

## What is deliberately *not* a data source

**tracker.gg.** Their developer program covers Apex, CS, Division 2, and
Splitgate — not VALORANT. TRN state they are not permitted to grant VALORANT API
access and redirect developers to Riot. Tracker Score is therefore unavailable
by any legitimate route. Scraping their site violates their ToS and would break
constantly. This is why the project builds its own rating metric instead.

**Riot's official VALORANT API.** Riot does not issue personal keys for
VALORANT. Production keys require a working prototype, RSO integration, and a
review that can run three weeks. Worth applying for *after* Phase 8, when a
prototype exists to show them — not before.
