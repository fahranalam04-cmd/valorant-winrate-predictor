"""Human-readable output.

Two audiences. The detailed view explains one scenario -- who is playing, what
moved, what the model said, whether the mirror held. The batch table ranks
everything at once and surfaces the extremes, because in a 343-scenario run the
interesting rows are the outliers, not the average.

Directional expectations print as `MODEL WARNING`, never as failures. A trained
model disagreeing with a reasonable assumption is a result worth reading, not a
scenario to retune until it agrees.
"""

from __future__ import annotations

from valwr.sandbox.schema import MatchScenario, ScenarioResult, VarianceResult

MIRROR_TOLERANCE = 1e-3          # see docs/SANDBOX.md for why this is not zero

# --- plain English ----------------------------------------------------
# Feature names like `d_acs_max` are precise and unreadable. Everything the
# report shows a human gets translated; the raw name stays available in the
# benchmark JSON for anyone who wants it.

STAT_NAMES = {
    "rating": "rating", "wr": "win rate", "tier": "rank",
    "acs": "combat score", "adr": "damage per round",
    "kast": "consistency (KAST)", "wr_map": "win rate on this map",
    "wr_agent": "win rate on this agent",
    "wr_map_agent": "win rate on this map+agent",
    "wr_recent": "recent form", "games_played": "games played",
    "games_map": "games on this map", "games_agent": "games on this agent",
    "games_map_agent": "games on this map+agent",
    "account_level": "account level", "fb_rate": "first bloods",
    "fd_rate": "first deaths", "rating_trend": "improving or declining",
    "days_since_last": "days since last played", "rating_n": "rated games",
}
AGG_NAMES = {"mean": "team average", "max": "best player's",
             "min": "worst player's", "std": "spread of"}
WHOLE_TEAM = {
    "n_duelist": "duelists", "n_controller": "controllers",
    "n_initiator": "initiators", "n_sentinel": "sentinels",
    "has_duelist": "has a duelist", "has_controller": "has a controller",
    "has_initiator": "has an initiator", "has_sentinel": "has a sentinel",
    "role_balance": "role balance", "n_off_role": "players off-role",
    "max_party": "largest party", "n_parties": "number of parties",
    "n_grouped": "players queued together",
    "n_with_history": "players we have history for",
}


def friendly(name: str) -> str:
    """Turn `d_acs_max` into "best player's combat score"."""
    key = name[2:] if name.startswith("d_") else name
    if key in WHOLE_TEAM:
        return WHOLE_TEAM[key]
    for suffix, agg in AGG_NAMES.items():
        if key.endswith("_" + suffix):
            stat = key[: -len(suffix) - 1]
            return f"{agg} {STAT_NAMES.get(stat, stat)}"
    return STAT_NAMES.get(key, key)


def verdict(p: float) -> str:
    """A plain reading of a probability, so nobody has to interpret 0.531."""
    edge = abs(p - 0.5)
    side = "Team A" if p > 0.5 else "Team B"
    if edge < 0.015:
        return "too close to call"
    if edge < 0.05:
        return f"{side} slightly ahead"
    if edge < 0.12:
        return f"{side} favoured"
    if edge < 0.25:
        return f"{side} clearly favoured"
    return f"{side} heavily favoured"




def _roster_lines(team, label: str) -> list[str]:
    names = [p.name for p in team.players]
    if len(set(names)) == 1:
        return [f"  5 x {names[0]}"]
    return [f"  P{i+1}  {n:<24} {team.agent_for(i)}"
            for i, n in enumerate(names)]


def scenario_detail(scenario: MatchScenario, result: ScenarioResult,
                    variance: VarianceResult | None = None) -> str:
    title = scenario.name.upper().replace("_", " ")
    out = [title, "=" * len(title), "",
           f"Category: {scenario.category}    Map: {scenario.map_name}",
           f"Model:    {result.model}", "",
           scenario.description, "",
           "Team A", "------", *_roster_lines(scenario.team_a, "A"), "",
           "Team B", "------", *_roster_lines(scenario.team_b, "B"), ""]

    movers = [(k, v) for k, v in sorted(result.features.items(),
                                        key=lambda kv: -abs(kv[1]))[:6]
              if abs(v) > 1e-9]
    if movers:
        out += ["Where the teams differ most:"]
        out += [f"  Team {'A' if v > 0 else 'B'} ahead on "
                f"{friendly(k):<36} ({abs(v):,.2f})"
                for k, v in movers]
        out.append("")

    p = result.probability
    out += ["Prediction:",
            f"  Team A  {p * 100:5.1f}%     {'#' * int(round(p * 40))}",
            f"  Team B  {(1 - p) * 100:5.1f}%     {'#' * int(round((1 - p) * 40))}",
            f"  -> {verdict(p)}", ""]

    if result.mirror_probability is not None:
        err = result.mirror_error
        symmetric = "ok" if err <= MIRROR_TOLERANCE else "FAILED"
        out += ["Mirror (sides swapped):",
                f"  {result.mirror_probability * 100:5.1f}% / "
                f"{(1 - result.mirror_probability) * 100:5.1f}%",
                f"  swapping the sides gives the mirror image "
                f"(off by {err * 100:.3f} points) -- {symmetric}", ""]

    if result.factors:
        out += ["What drove the prediction:"]
        out += [f"  {'towards A' if v > 0 else 'towards B'}  "
                f"{friendly(k):<38} ({abs(v):.3f})"
                for k, v in result.factors]
        out.append("")

    if scenario.expect:
        met = result.expectation_met
        if met is None:
            state = "not evaluable"
        else:
            state = "PASS" if met else "MODEL WARNING -- model disagrees"
        out += [f"Scenario expectation: {scenario.expect}", f"Result: {state}", ""]
    elif "observational" in scenario.tags:
        out += ["Scenario expectation: none (observational -- reported, not judged)", ""]

    if variance is not None:
        out += variance_detail(variance).splitlines()
    return "\n".join(out)


def variance_detail(v: VarianceResult) -> str:
    return "\n".join([
        f"Variance ({v.samples} samples, seed {v.seed}):",
        f"  static      {v.static_probability * 100:5.1f}%",
        f"  mean        {v.mean * 100:5.1f}%   (drift {v.drift * 100:+.1f} pts)",
        f"  median      {v.median * 100:5.1f}%",
        f"  std dev     {v.std * 100:5.2f} pts",
        f"  range       {min(v.probabilities) * 100:5.1f}% .. "
        f"{max(v.probabilities) * 100:.1f}%",
        f"  5/25/75/95  {v.percentile(0.05) * 100:.1f}% / "
        f"{v.percentile(0.25) * 100:.1f}% / {v.percentile(0.75) * 100:.1f}% / "
        f"{v.percentile(0.95) * 100:.1f}%",
        f"  P>0.50      {sum(1 for p in v.probabilities if p > 0.5) / v.samples:.1%}"
        f"   P>0.60 {sum(1 for p in v.probabilities if p > 0.6) / v.samples:.1%}"
        f"   P<0.40 {sum(1 for p in v.probabilities if p < 0.4) / v.samples:.1%}",
        f"  favourite flips {v.flip_rate:.1%}",
        f"  robustness  {v.robustness}",
        "",
    ])


def batch_table(results: list[ScenarioResult],
                variances: dict[str, VarianceResult] | None = None) -> str:
    """One row per scenario, sorted by how lopsided the prediction is.

    Columns appear only when there is data for them -- an earlier version
    printed three columns of `nan` whenever variance had not been run, which
    made the table look broken.
    """
    variances = variances or {}
    show_noise = bool(variances)

    head = f"  {'scenario':<32}{'A vs B':>11}   {'reading':<28}"
    if show_noise:
        head += f"{'noise':>8}{'flips':>8}  "
    head += "check"
    lines = [head, "  " + "-" * (len(head) - 2)]

    for r in sorted(results, key=lambda r: -abs(r.probability - 0.5)):
        met = r.expectation_met
        if r.mirror_error is not None and r.mirror_error > MIRROR_TOLERANCE:
            check = "ASYMMETRIC"
        elif met is None:
            check = "-"
        elif met:
            check = "ok"
        else:
            check = "UNEXPECTED"

        p = r.probability
        row = (f"  {r.scenario[:31]:<32}"
               f"{p * 100:>3.0f}% / {(1 - p) * 100:>3.0f}%   "
               f"{verdict(p):<28}")
        if show_noise:
            v = variances.get(r.scenario)
            row += (f"{('+/-' + format(v.std * 100, '.1f')) if v else '':>8}"
                    f"{(format(v.flip_rate, '.0%') if v else ''):>8}  ")
        lines.append(row + check)

    lines += ["", "  reading is just the probability in words. "
                  "check: ok = matched the",
              "  scenario's expectation, UNEXPECTED = the model disagreed "
              "(a finding, not",
              "  a failure), - = no expectation was claimed, ASYMMETRIC = "
              "swapping the",
              "  two teams did not mirror the prediction."]
    if show_noise:
        lines.append("  noise = how much the prediction moves under realistic "
                     "variation; flips =")
        lines.append("  how often the favourite changes side.")
    return chr(10).join(lines)


def highlights(results: list[ScenarioResult],
               variances: dict[str, VarianceResult] | None = None) -> str:
    variances = variances or {}
    if not results:
        return "no results"
    out = ["", "Highlights", "----------"]

    strongest = max(results, key=lambda r: abs(r.probability - 0.5))
    closest = min(results, key=lambda r: abs(r.probability - 0.5))
    out += [f"  strongest prediction  {strongest.scenario} "
            f"({strongest.probability * 100:.1f}%)",
            f"  closest to even       {closest.scenario} "
            f"({closest.probability * 100:.1f}%)"]

    with_mirror = [r for r in results if r.mirror_error is not None]
    if with_mirror:
        worst = max(with_mirror, key=lambda r: r.mirror_error)
        out.append(f"  largest mirror error  {worst.scenario} "
                   f"({worst.mirror_error:.6f})")

    if variances:
        vs = list(variances.values())
        out += [f"  most sensitive        "
                f"{max(vs, key=lambda v: v.std).scenario} "
                f"(std {max(v.std for v in vs) * 100:.2f} pts)",
                f"  most robust           "
                f"{min(vs, key=lambda v: v.std).scenario}",
                f"  largest drift         "
                f"{max(vs, key=lambda v: abs(v.drift)).scenario} "
                f"({max(vs, key=lambda v: abs(v.drift)).drift * 100:+.1f} pts)",
                f"  highest flip rate     "
                f"{max(vs, key=lambda v: v.flip_rate).scenario} "
                f"({max(v.flip_rate for v in vs):.1%})"]

    warned = [r for r in results if r.expectation_met is False]
    out += ["", f"  directional expectations violated: {len(warned)}"]
    for r in warned[:12]:
        out.append(f"    {r.scenario:<34} expected {r.expect}, "
                   f"got {r.probability * 100:.1f}%")
    if len(warned) > 12:
        out.append(f"    ... and {len(warned) - 12} more")
    return "\n".join(out)
