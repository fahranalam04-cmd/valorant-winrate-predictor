"""Execute scenarios through the production feature path.

Every scenario is run twice -- as written and mirrored -- because the mirror is
generated rather than hand-written. Two hand-maintained copies of the same
match would eventually drift apart, and the drift would look like a model
finding.

Nothing here reimplements a feature. `features.build.build_match` is called
with `require_outcome=False`, the same flag the live client uses, so the
sandbox and the live path produce features through identical code.
"""

from __future__ import annotations

import math

from valwr.features import build as fb
from valwr.sandbox import world
from valwr.sandbox.predictor import Predictor, linear_contributions
from valwr.sandbox.schema import MatchScenario, ScenarioResult

# A fixed instant so static scenarios are reproducible forever. Any constant
# works; this one is simply a round number well clear of the collected data.
AS_OF = 1_800_000_000


def features_for(scenario: MatchScenario, bundle: dict,
                 as_of: int = AS_OF) -> dict[str, float]:
    """The 52 production difference features for a scenario."""
    scenario.validate()
    conn, roster = world.build_world(scenario, as_of)
    try:
        match = {
            "match_id": f"sandbox:{scenario.name}",
            "started_at": as_of,
            "map": scenario.map_name,
            "season": "sandbox",
            "region": "na",
            "winner": None,            # it has not been played
            "rounds_blue": None,
            "rounds_red": None,
        }
        mf = fb.build_match(conn, match, roster, bundle["norms"],
                            bundle["prior_rate"], bundle["roles"],
                            require_outcome=False)
        if mf is None:
            raise RuntimeError(f"{scenario.name}: build_match returned None")
        for key, value in mf.values.items():
            if not math.isfinite(value):
                raise ValueError(f"{scenario.name}: feature {key} is {value}")
        return dict(mf.values)
    finally:
        conn.close()


def run(scenario: MatchScenario, predictor: Predictor, bundle: dict,
        as_of: int = AS_OF, with_mirror: bool = True) -> ScenarioResult:
    features = features_for(scenario, bundle, as_of)
    probability = predictor.predict_proba(features)

    mirror_p = None
    if with_mirror:
        mirror_features = features_for(scenario.mirrored(), bundle, as_of)
        mirror_p = predictor.predict_proba(mirror_features)

    return ScenarioResult(
        scenario=scenario.name,
        category=scenario.category,
        model=predictor.name,
        probability=probability,
        features=features,
        mirror_probability=mirror_p,
        expect=scenario.expect,
        factors=tuple(linear_contributions(bundle, features)),
    )


def run_many(scenarios, predictor: Predictor, bundle: dict,
             as_of: int = AS_OF, with_mirror: bool = True
             ) -> list[ScenarioResult]:
    return [run(s, predictor, bundle, as_of, with_mirror) for s in scenarios]


def run_variance(scenario: MatchScenario, predictor: Predictor, bundle: dict,
                 samples: int = 1000, seed: int = 42, as_of: int = AS_OF):
    """Monte Carlo around a scenario. Returns a VarianceResult."""
    from valwr.sandbox import variance
    from valwr.sandbox.schema import VarianceResult

    static = predictor.predict_proba(features_for(scenario, bundle, as_of))
    draws = []
    for realisation in variance.sample(scenario, samples, seed):
        draws.append(predictor.predict_proba(
            features_for(realisation, bundle, as_of)))
    return VarianceResult(
        scenario=scenario.name, model=predictor.name, samples=samples,
        seed=seed, static_probability=static, probabilities=tuple(draws))


def antisymmetry_error(scenario: MatchScenario, bundle: dict,
                       as_of: int = AS_OF) -> float:
    """Largest |f(A,B) + f(B,A)| across the raw difference features.

    This should be exactly zero: the representation is a difference, so
    swapping sides must negate every column. Unlike the *probability* mirror
    -- which the scaler's non-zero means push off by ~3e-4 -- there is no
    numerical excuse here beyond floating point.
    """
    a = features_for(scenario, bundle, as_of)
    b = features_for(scenario.mirrored(), bundle, as_of)
    return max((abs(a[k] + b.get(k, 0.0)) for k in a), default=0.0)
