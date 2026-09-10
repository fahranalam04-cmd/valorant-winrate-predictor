# Ethics and Terms of Service

This is a ban-safety document. The rules here are not stylistic preferences —
several of them are the difference between a tolerated tool and a permanently
banned account.

---

## The local client API

Reading the VALORANT local client API is what every rank-checker overlay does.
It is widely tolerated. The community summary is "as long as you use common
sense and don't do anything a Riot employee would frown at, you won't get
banned" — which is accurate but unhelpfully vague. Concretely:

### Allowed — what this project does

- Read the lockfile to authenticate against `127.0.0.1`
- Read match state: the current match ID, the ten PUUIDs, locked agents, teams
- Read your own account identity and region
- Poll for the current match every few seconds (the client's websocket would
  be equally acceptable; it is simply not needed)
- Display information in a **separate browser window**, or on your own phone

### Forbidden — what this project must never do

- **Never POST or write to the client API.** Read-only, without exception.
- **Never automate agent selection.** Instalockers are the single most common
  cause of bans in this category of tool. Do not select, lock, hover, or dodge.
- **Never read or write process memory.** No injection, no DLLs, no hooks.
- **Never draw an in-game overlay.** The dashboard is a browser page on a second
  monitor or behind alt-tab. Overlay injection into an anti-cheat-protected
  process is exactly what Vanguard is built to catch.
- **Never automate any gameplay action** — no aim assistance, no input
  synthesis, no queue manipulation.

The line is simple: **observe, never act.** Everything on the forbidden list
involves the program taking an action inside the game. Everything on the allowed
list is reading state that the client already has.

## Third-party API use

- Respect HenrikDev's published rate limits. Back off on `429` and honour
  `Retry-After`.
- **Do not evade limits.** No key rotation, no proxy pools, no parallel
  accounts. The API is a free service run by one person; treating it as
  infrastructure to be maximally extracted from is how free services die.
- Describe the project honestly when requesting a key. It is a personal
  educational ML project. Say that.
- Cache aggressively. Every avoided call is the polite outcome.

## Other players' data

The crawler collects real people's match histories. They did not consent to
being in a training set, even though the data is publicly visible.

- **The database never leaves this machine.** `data/` is gitignored.
- **Never publish a dataset** of PUUIDs, Riot IDs, or per-player statistics.
- **Never expose a public endpoint** that looks up an arbitrary player. The
  dashboard serves only the match you are in, has no route that takes a player,
  and binds to `127.0.0.1`. `phone.bat` widens that to your own network — opt-in,
  announced when it starts, and still refusing requests from other sites. It is
  never for the internet. See [SECURITY.md](../SECURITY.md).
- **Party markers are inferred, locally, for your lobby only.** The client does
  not reveal the enemy's party, so the dashboard infers it from match history
  already on this machine: two players who have queued together before. It is
  computed for the ten players in front of you and cannot be searched.
- **Aggregate freely, identify never.** Model weights, feature importances, and
  distribution statistics are fine to publish. Individual rows are not.
- **Every published image and the public demo use invented players**, from
  `valwr/dash/demo.py`. No real Riot ID appears in a screenshot, and none may.

Note that HenrikDev's leaderboard exposes `is_anonymized` — players who opted
out of public leaderboard identification. Skip them entirely.

## Secrets

- `.env` is gitignored; only `.env.example` is tracked.
- No key literal in any committed file, test fixture, or notebook output.
  Notebook outputs in particular leak keys and player data — clear them before
  committing, or do not commit notebooks.
- Before the first push, and periodically after:
  ```bash
  git log -p | grep -iE "HDEV-|sk-ant-"
  ```
  must return nothing.

## Honesty about results

This is a portfolio project, so the results are a claim about your competence.

- Never report a metric that was not measured on a held-out, time-split test set.
- Never populate the README results table with an aspirational number.
- If a model scores suspiciously well, treat it as a leakage bug until proven
  otherwise. `docs/MODELING.md` lists the traps; `test/test_leakage.py` is the
  enforcement.
- State the limitations in the README. A project that knows its own ceiling
  reads as more competent than one claiming an implausible result — and an
  interviewer who finds the leakage you missed is a much worse outcome than one
  reading that you found it yourself.

## Attribution

Not affiliated with or endorsed by Riot Games. VALORANT is a trademark of Riot
Games, Inc. Data via the unofficial HenrikDev API and valorant-api.com, neither
of which is affiliated with Riot either.

Agent and map artwork is Riot's. It is downloaded locally by
`tools/fetch_agent_art.py` and never committed to the main branch; the public
demo and the README screenshots show it under Riot's terms for free fan
projects, with this attribution.
