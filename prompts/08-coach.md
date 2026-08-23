# Phase 8 — Claude coach

**Load the `claude-api` skill first** — model IDs, pricing, and SDK patterns
should come from there rather than from memory.

Read `CLAUDE.md` before starting. Phase 7 must pass.

## Goal

Turn the prediction and its attributions into coaching advice and a plain-English
explanation of *why* the model predicted what it did.

## The rule this phase lives or dies by

**The LLM explains the model. It does not invent analysis.**

It receives numbers — the prediction, SHAP attributions, both comps, per-player
summaries — and may reason only over those. It must never generate a statistic
that was not supplied.

This matters more than it sounds. An LLM asked about VALORANT will happily
produce confident, specific, entirely fabricated claims — "Jett has a 63% win
rate on Ascent B site" — that read exactly like the real numbers next to them. A
coach that does that is worse than no coach, because it is convincing, and it
poisons the credibility of the real model output surrounding it.

Grounding is the interesting engineering here, and it is the part worth writing
about.

## Build

**1. Payload — `valwr/coach/payload.py`**

Assemble a structured input: prediction and confidence, top SHAP attributions
with human-readable feature names, both team comps with roles, per-player
summaries (rank, rating, relevant map/agent history), map, and which players
were unresolved.

Keep it compact and explicit. Every number the model is allowed to cite must be
in here, labelled. Anything not in the payload is not citable.

**2. Prompting — `valwr/coach/prompt.py`**

System prompt must establish:

- Role: explain a statistical model's output to a player about to start a match
- **Hard constraint: use only the numbers provided.** No statistics from
  training data, no general VALORANT knowledge presented as data about these
  players. If something is not in the payload, it cannot be stated as fact.
- Distinguish clearly between what the model measured and what is general
  strategic advice. General tactical reasoning about a comp is fine and useful —
  it just must not be dressed up as a statistic.
- Acknowledge uncertainty: a 52% prediction is nearly a coin flip and should
  read that way.
- Tone: concise and direct. A player has ~30 seconds in agent select.

**3. Two outputs — `valwr/coach/coach.py`**

- **Explanation**: why the model predicted this, in plain English, referencing
  the actual attributions
- **Advice**: what this team might do about it, given the comps and the map

**4. Cost control**

One call per match, cached by `match_id`. Never call on every websocket event —
agent select fires many. Log token usage so the per-match cost is visible.

**5. Grounding check — `test/test_coach.py`**

The test that matters. Take a sample of outputs and verify every numeric claim
appears in the input payload. A regex sweep for numbers, cross-referenced
against payload values, catches most fabrication.

Also test the degraded case: with several players unresolved, the output should
acknowledge the missing data rather than confidently analysing players it knows
nothing about.

**6. Dashboard integration**

Add to the Phase 7 page. Visually distinct from the model's own output — it
should be clear which text is the model and which is the LLM interpreting it.

## Constraints

- `ANTHROPIC_API_KEY` from `.env`. Never committed.
- Handle API failures gracefully — the dashboard still works without the coach.
  It is a layer on top, not a dependency.
- Never send PUUIDs or Riot IDs to the API. Anonymise to "Player 1..5" in the
  payload. Other players did not consent to their identities going to a
  third-party service; see `docs/ETHICS-AND-TOS.md`. The advice does not need
  their names.

## Acceptance criteria

1. Given a real match, returns coherent advice and explanation
2. **Every numeric claim traces back to the input payload** — verify on a sample
   by hand, not just via the test
3. Degraded case acknowledges missing players
4. One cached call per match; cost per match reported
5. No identifying player data leaves the machine

## Then

The project is functionally complete. Consider Phase 9: the backtest harness,
the README results writeup, and — now that a prototype exists to show them —
applying to Riot for a production key and RSO.
