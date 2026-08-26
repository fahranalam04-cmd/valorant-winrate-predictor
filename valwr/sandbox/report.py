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
        out += ["Largest feature differences (A - B):"]
        out += [f"  {'+' if v > 0 else '-'} {k.replace('d_', ''):<26} {abs(v):.4f}"
                for k, v in movers]
        out.append("")

    p = result.probability
    out += ["Prediction:",
            f"  Team A: {p * 100:5.1f}%",
            f"  Team B: {(1 - p) * 100:5.1f}%", ""]

    if result.mirror_probability is not None:
        err = result.mirror_error
        verdict = "PASS" if err <= MIRROR_TOLERANCE else "FAIL"
        out += ["Mirror (sides swapped):",
                f"  {result.mirror_probability * 100:5.1f}% / "
                f"{(1 - result.mirror_probability) * 100:5.1f}%",
                f"  |p + p_mirror - 1| = {err:.6f}   {verdict}", ""]

    if result.factors:
        out += ["Top model factors (linear contributions):"]
        out += [f"  {'+' if v > 0 else '-'} {k.replace('d_', ''):<26} {abs(v):.4f}"
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
    variances = variances or {}
    head = (f"  {'scenario':<34}{'static':>8}{'var mean':>10}{'std':>7}"
            f"{'flip%':>7}{'mirror':>9}  status")
    lines = [head, "  " + "-" * (len(head) - 2)]
    for r in sorted(results, key=lambda r: -abs(r.probability - 0.5)):
        v = variances.get(r.scenario)
        met = r.expectation_met
        status = ("PASS" if met else "MODEL WARNING") if met is not None else "--"
        if r.mirror_error is not None and r.mirror_error > MIRROR_TOLERANCE:
            status = "MIRROR FAIL"
        lines.append(
            f"  {r.scenario[:33]:<34}{r.probability * 100:>7.1f}%"
            f"{(v.mean * 100 if v else float('nan')):>9.1f}%"
            f"{(v.std * 100 if v else float('nan')):>6.2f}"
            f"{(v.flip_rate * 100 if v else float('nan')):>6.1f}%"
            f"{(r.mirror_error or 0):>9.5f}  {status}")
    return "\n".join(lines)


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
