"""Export the whole static catalog (plus light variance) for the dashboard."""
import json, pathlib, sys
sys.path.insert(0, ".")
from valwr.sandbox import predictor as pred, report, runner, scenarios

bundle = pred.load_bundle()
models = {p.name: p for p in pred.predictors(bundle, "logistic,gbm")}
lin, gbm = models["logistic"], models["gbm"]

out = {"model": bundle.get("best"), "scenarios": [], "categories": {}}
CAT_BLURB = {
 "sanity": "Controls. If these look wrong, nothing else can be trusted.",
 "skill": "One side is simply better, by graded amounts.",
 "carry": "One outstanding player among ordinary ones -- smurfs and hard carries.",
 "weak_link": "One liability dragging an otherwise fine team down.",
 "rank": "Rank gaps, and cases where rank disagrees with actual performance.",
 "map": "Map comfort: teams and individuals who are strong or weak here.",
 "agent": "Agent comfort, including players forced onto agents they are bad with.",
 "map_agent": "The interaction on its own -- fine on the map, fine on the agent, special together.",
 "shrinkage": "Small samples versus large ones. A 3-game streak should not outweigh 500 games.",
 "form": "Hot and cold streaks, and streaks that contradict a career record.",
 "rust": "Time since last played, from today to months idle.",
 "experience": "Account age and games played, deliberately separated from skill.",
 "composition": "Team compositions: missing roles, stacked roles, balance versus raw skill.",
 "off_role": "Players on agents outside the role they normally play.",
 "party": "Queue structure, from five solos to a full five-stack.",
 "distribution": "Same average strength, different shape -- one star versus five even players.",
 "coverage": "How many players we actually have history for, including none at all.",
 "contradiction": "Signals that disagree. Real matches rarely point one way.",
 "boundary": "Extreme but legal values -- 0% and 100% rates, rank floors and ceilings.",
 "dominance": "One side better on essentially everything, in graded amounts.",
 "cancellation": "Advantages deliberately set to offset each other.",
}

for s in scenarios.CATALOG:
    r = runner.run(s, lin, bundle)
    g = runner.run(s, gbm, bundle, with_mirror=True)
    v = runner.run_variance(s, lin, bundle, samples=24, seed=42)
    diffs = sorted(r.features.items(), key=lambda kv: -abs(kv[1]))[:6]
    out["scenarios"].append({
        "name": s.name, "category": s.category, "description": s.description,
        "map": s.map_name, "expect": s.expect,
        "observational": "observational" in s.tags,
        "teamA": [{"n": p.name, "a": s.team_a.agent_for(i)}
                  for i, p in enumerate(s.team_a.players)],
        "teamB": [{"n": p.name, "a": s.team_b.agent_for(i)}
                  for i, p in enumerate(s.team_b.players)],
        "p": round(r.probability, 4),
        "pGbm": round(g.probability, 4),
        "mirrorErr": round(r.mirror_error or 0, 6),
        "mirrorErrGbm": round(g.mirror_error or 0, 6),
        "met": r.expectation_met,
        "reading": report.verdict(r.probability),
        "diffs": [{"f": report.friendly(k), "v": round(v_, 3)} for k, v_ in diffs],
        "factors": [{"f": report.friendly(k), "v": round(v_, 4)}
                    for k, v_ in r.factors],
        "var": {"mean": round(v.mean, 4), "std": round(v.std, 4),
                "lo": round(v.percentile(0.05), 4),
                "hi": round(v.percentile(0.95), 4),
                "flip": round(v.flip_rate, 3), "robust": v.robustness},
    })
    print(".", end="", flush=True)

for name, n in scenarios.categories().items():
    out["categories"][name] = {"n": n, "blurb": CAT_BLURB.get(name, "")}

path = pathlib.Path("reports/sandbox/dashboard_data.json")
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  {path.stat().st_size/1024:.0f} KB, "
      f"{len(out['scenarios'])} scenarios")
