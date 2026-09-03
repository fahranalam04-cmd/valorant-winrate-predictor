"""Terminal rendering of a live poll.

Takes the dictionary `live/state.poll_once` returns and prints it. Computes
nothing: if a number is not in the state, it does not belong on screen. The
browser dashboard renders the same dictionary, which is the only way to be sure
the two views cannot drift apart.
"""

from __future__ import annotations

import unicodedata

NAME_WIDTH = 20


def display_width(s: str) -> int:
    """Terminal columns a string occupies, not its character count.

    Korean and Japanese gamertags are common in this data and render two
    columns per glyph, so `f"{name:<20}"` under-pads them and the table's
    columns walk out of line.
    """
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
               for c in s)


def fit(s: str, width: int) -> str:
    """Truncate to `width` display columns, then pad to exactly that."""
    if display_width(s) > width:
        out = ""
        for ch in s:
            if display_width(out + ch) > width - 1:
                break
            out += ch
        s = out + "…"
    return s + " " * max(0, width - display_width(s))


def _header(state: dict) -> None:
    kind = "CUSTOM" if state["is_custom"] else state["phase"].upper()
    sizes = state["team_sizes"]
    # In pregame Riot exposes only AllyTeam, so the enemy count is structurally
    # zero. Printing "5v0" there reads as if the other side vanished.
    if state["phase"] == "pregame":
        size = (f"{sizes['Blue'] + sizes['Red']} on your team, "
                f"enemy hidden until the match starts")
    else:
        size = f"{sizes['Blue']}v{sizes['Red']}"
    print("\n" + "=" * 58)
    print(f"  {kind}  ·  {state['map'] or 'unknown map'}"
          f"  ·  {state['mode'] or 'unknown mode'}  ·  {size}")
    print("=" * 58)
    for w in state["warnings"]:
        print(f"  {w}")
    if state["warnings"]:
        print()


def _prediction(state: dict) -> None:
    p = state["prediction"]
    if p is None:
        print("  not enough of the roster to predict yet")
        return
    own = p["own_probability"]
    bar = int(round(own * 30))
    print(f"\n  YOUR TEAM ({state['own_team']})   {own * 100:5.1f}%")
    print(f"  {'#' * bar}{'-' * (30 - bar)}")
    print(f"  ENEMY                {(1 - own) * 100:5.1f}%")
    print(f"\n  {state['coverage']}/10 players known "
          f"({state['fetched']} fetched live) -- confidence "
          f"{state['confidence']}")
    print(f"  model: {state['model']}")
    if p["factors"]:
        print("\n  strongest factors:")
        for f in p["factors"]:
            arrow = "+" if f["value"] > 0 else "-"
            name = f["name"].replace("d_", "")
            print(f"    {arrow} {name:<26} {abs(f['value']):.3f}")


def _side(state: dict, team: str, title: str) -> int:
    rows = [p for p in state["players"] if p["team"] == team]
    if not rows:
        return 0
    print(f"  {title}".ljust(NAME_WIDTH + 22) + "potential")
    print("  " + "-" * (NAME_WIDTH + 34))
    flagged = 0
    for i, p in enumerate(rows, 1):
        mark = "*" if p["is_you"] else " "
        label = fit(p["name"], NAME_WIDTH)
        if p["score"] is None:
            print(f"  {i}{mark} {label} {p['agent']:<10} {'--':>3}   no history")
            continue
        bang = " !" if p["flag"] else "  "
        print(f"  {i}{mark} {label} {p['agent']:<10} "
              f"{p['score']:>3}{bang}  {p['reason']}")
        if p["flag"]:
            flagged += 1
            print(f"       {' ' * NAME_WIDTH} {'':<10}      ^ {p['flag']['note']}")
    return flagged


def _teams(state: dict, top1_rate: float | None) -> None:
    flagged = _side(state, state["own_team"], f"YOUR TEAM ({state['own_team']})")
    enemy = [p for p in state["players"] if p["team"] == state["enemy_team"]]
    if enemy:
        print()
        flagged += _side(state, state["enemy_team"],
                         f"ENEMY ({state['enemy_team']})")

    print("\n  * you. Score is a percentile: 70 means likely to outperform "
          "70% of players.")
    # Measured figure travels with the index that produced it. A literal here
    # went silently wrong at the next retrain, with nothing to catch it.
    if top1_rate is not None:
        print(f"  Ranks the top player correctly {top1_rate * 100:.1f}% of the "
              f"time against 20%\n  chance -- a real edge, not a reliable one. "
              f"See docs/MODEL-CHOICE.md.")
    else:
        print("  A real edge over chance, not a reliable one. "
              "See docs/MODEL-CHOICE.md.")
    if flagged:
        print(f"\n  ! {flagged} player(s) performing above their rank AND "
              f"topping their lobbies\n    far more often than one game in "
              f"five. That is what a smurf looks like,\n    but it also fits a "
              f"returning player or someone mid-climb -- it is a\n    flag, "
              f"not an accusation.")


def show(state: dict, top1_rate: float | None = None) -> None:
    """Print one poll."""
    _header(state)
    _prediction(state)
    if state["players"]:
        print()
        _teams(state, top1_rate)
    print()
