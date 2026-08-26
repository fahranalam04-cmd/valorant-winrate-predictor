"""The curated scenario catalog.

Human-readable VALORANT situations, organised by what each one probes. Layers
2-4 (per-feature sweeps, pairwise grids, context sweeps) are *generated* in
sweeps.py rather than written here, and every scenario's mirror is generated in
schema.MatchScenario.mirrored() -- writing either by hand would create copies
that drift apart.

`expect` is a directional claim, and it is reported rather than enforced. If
the trained model disagrees with a reasonable assumption, that is a finding
about the model, not a scenario to quietly retune.
"""

from __future__ import annotations

from valwr.sandbox import profiles as P
from valwr.sandbox.schema import MatchScenario, TeamProfile, VarianceSpec

# A role-balanced composition, used wherever composition is not the variable
# under test so it cannot confound anything.
BALANCED = ("Jett", "Omen", "Sova", "Killjoy", "Raze")
ROLE_OF = {"Jett": "Duelist", "Raze": "Duelist", "Phoenix": "Duelist",
           "Omen": "Controller", "Brimstone": "Controller", "Astra": "Controller",
           "Sova": "Initiator", "Breach": "Initiator", "Skye": "Initiator",
           "Killjoy": "Sentinel", "Cypher": "Sentinel", "Sage": "Sentinel"}
MAPS = ("Ascent", "Bind", "Haven", "Split", "Icebox", "Lotus", "Sunset")


def team(*players, agents=BALANCED) -> TeamProfile:
    if len(players) == 1:
        players = players * 5
    return TeamProfile(players=tuple(players), agents=tuple(agents))


def swap(base, index: int, replacement) -> tuple:
    """A five-player roster with one slot replaced."""
    roster = list(base if isinstance(base, (list, tuple)) else [base] * 5)
    if len(roster) == 1:
        roster = roster * 5
    roster[index] = replacement
    return tuple(roster)


def scenario(name, category, description, a, b, *, map_name="Ascent",
             expect=None, tags=(), variance=None) -> MatchScenario:
    return MatchScenario(
        name=name, category=category, description=description,
        team_a=a, team_b=b, map_name=map_name, expect=expect,
        tags=tuple(tags), variance=variance or VarianceSpec())


def _keeps_strongest_first(sc: MatchScenario) -> bool:
    """Variance guard: the carry must remain the best player on their team."""
    ratings = [p.acs for p in sc.team_a.players]
    return ratings[2] == max(ratings)


def _keeps_weakest(sc: MatchScenario) -> bool:
    ratings = [p.acs for p in sc.team_a.players]
    return ratings[2] == min(ratings)


def build() -> list[MatchScenario]:
    s: list[MatchScenario] = []
    avg, weak, strong, elite = P.AVERAGE, P.WEAK, P.STRONG, P.ELITE
    above, below = P.ABOVE_AVERAGE, P.BELOW_AVERAGE

    # --- 1. neutral / sanity -----------------------------------------
    s += [
        scenario("fair_match", "sanity",
                 "Five identical average players on each side",
                 team(avg), team(avg), expect="even"),
        scenario("fair_heterogeneous", "sanity",
                 "Different internal distributions, comparable aggregate strength",
                 team(*(strong, avg, avg, avg, weak)),
                 team(*(above, above, avg, below, below)), expect="even"),
        scenario("fair_experienced_vs_new", "sanity",
                 "Long histories against short ones at the same skill",
                 team(P.EXPERIENCED_AVERAGE), team(P.NEW_PLAYER)),
    ]

    # --- 2. graded skill imbalance ------------------------------------
    for label, other, exp in (("tiny", P.ABOVE_AVERAGE, "b_favoured"),
                              ("small", P.STRONG, "b_favoured"),
                              ("moderate", P.ELITE, "b_favoured")):
        s.append(scenario(f"skill_edge_{label}", "skill",
                          f"Five average against five {other.name}",
                          team(avg), team(other), expect=exp))
    s += [
        scenario("skill_edge_large", "skill", "Weak against strong",
                 team(weak), team(strong), expect="b_favoured"),
        scenario("skill_edge_extreme", "skill", "Weak against elite",
                 team(weak), team(elite), expect="b_favoured"),
    ]

    # --- 3. smurf / carry ---------------------------------------------
    carry_var = VarianceSpec(preserve=_keeps_strongest_first)
    s += [
        scenario("single_smurf", "carry",
                 "One smurf among four average players, against five average",
                 team(*swap(avg, 2, P.SMURF_LIKE)), team(avg),
                 expect="a_favoured", variance=carry_var),
        scenario("smurf_carrying_weak", "carry",
                 "One smurf with four weak players against five average",
                 team(*swap(weak, 2, P.SMURF_LIKE)), team(avg),
                 variance=carry_var),
        scenario("elite_carrying_weak", "carry",
                 "One elite with four weak against five above-average",
                 team(*swap(weak, 2, elite)), team(above), variance=carry_var),
        scenario("two_carries", "carry",
                 "Two elite players and three weak, against five average",
                 team(*(elite, elite, weak, weak, weak)), team(avg)),
        scenario("carry_on_both_sides", "carry",
                 "Equal carries, different supporting casts",
                 team(*swap(avg, 2, elite)), team(*swap(weak, 2, elite)),
                 expect="a_favoured"),
        scenario("hidden_smurf", "carry",
                 "Low displayed tier, elite rating and performance",
                 team(*swap(avg, 2, P.SMURF_LIKE)), team(avg),
                 expect="a_favoured", variance=carry_var),
        scenario("rank_only_smurf", "carry",
                 "High rank, ordinary performance -- rank without substance",
                 team(*swap(avg, 2, P.RANK_ONLY_SMURF)), team(avg)),
    ]

    # --- 4. weak links -------------------------------------------------
    weak_var = VarianceSpec(preserve=_keeps_weakest)
    s += [
        scenario("single_weak_link", "weak_link",
                 "One weak player among four average, against five average",
                 team(*swap(avg, 2, weak)), team(avg), expect="b_favoured",
                 variance=weak_var),
        scenario("weak_link_among_strong", "weak_link",
                 "One weak player among four strong",
                 team(*swap(strong, 2, weak)), team(above)),
        scenario("two_weak_links", "weak_link", "Two weak players among three average",
                 team(*(avg, avg, avg, weak, weak)), team(avg),
                 expect="b_favoured"),
        scenario("one_unknown", "weak_link", "One player with no history at all",
                 team(*swap(avg, 2, P.UNKNOWN)), team(avg)),
        scenario("several_unknown", "weak_link", "Three players with no history",
                 team(*(avg, avg, P.UNKNOWN, P.UNKNOWN, P.UNKNOWN)), team(avg)),
        scenario("elite_plus_liability", "weak_link",
                 "One elite and one terrible among three average, versus five average",
                 team(*(elite, weak, avg, avg, avg)), team(avg)),
    ]

    # --- 5. rank -------------------------------------------------------
    def at_tier(profile, tier):
        return profile.with_(tier=tier, name=f"{profile.name}_t{tier}")

    s += [
        scenario("rank_identical", "rank", "Same rank, same statistics",
                 team(avg), team(avg), expect="even"),
        scenario("rank_tiny_gap", "rank", "One tier apart, statistics identical",
                 team(at_tier(avg, 16)), team(at_tier(avg, 15))),
        scenario("rank_one_tier", "rank", "Three tiers apart, statistics identical",
                 team(at_tier(avg, 18)), team(at_tier(avg, 15))),
        scenario("rank_many_tiers", "rank", "Nine tiers apart, statistics identical",
                 team(at_tier(avg, 24)), team(at_tier(avg, 15))),
        scenario("rank_spread_wide", "rank", "One team's ranks vary enormously",
                 team(*(at_tier(avg, 24), at_tier(avg, 21), at_tier(avg, 15),
                        at_tier(avg, 9), at_tier(avg, 6))), team(at_tier(avg, 15))),
        scenario("rank_one_high_among_low", "rank",
                 "A single high rank surrounded by much lower ones",
                 team(*swap(at_tier(avg, 9), 2, at_tier(avg, 25))),
                 team(at_tier(avg, 12))),
        scenario("rank_high_stats_low", "rank",
                 "Higher rank but worse performance -- a direct contradiction",
                 team(at_tier(below, 21)), team(at_tier(above, 12))),
        scenario("rank_low_stats_high", "rank",
                 "Lower rank but better performance -- the reverse contradiction",
                 team(at_tier(above, 12)), team(at_tier(below, 21))),
        scenario("rank_balanced_vs_mixed", "rank",
                 "Five similar ranks against a heterogeneous spread",
                 team(at_tier(avg, 15)),
                 team(*(at_tier(avg, 21), at_tier(avg, 18), at_tier(avg, 15),
                        at_tier(avg, 12), at_tier(avg, 9)))),
    ]

    # --- 6. map --------------------------------------------------------
    s += [
        scenario("map_neutral", "map", "Neither side has map history worth noting",
                 team(avg), team(avg), expect="even"),
        scenario("good_map", "map", "Every Team A player is strong on this map",
                 team(P.MAP_SPECIALIST), team(avg), expect="a_favoured"),
        scenario("bad_map", "map", "Every Team A player is weak on this map",
                 team(P.MAP_WEAK), team(avg), expect="b_favoured"),
        scenario("single_map_specialist", "map", "One player far stronger on this map",
                 team(*swap(avg, 2, P.MAP_SPECIALIST)), team(avg)),
        scenario("single_map_liability", "map", "One player far weaker on this map",
                 team(*swap(avg, 2, P.MAP_WEAK)), team(avg)),
        scenario("mixed_map_comfort", "map", "Two strong, two weak, one neutral on the map",
                 team(*(P.MAP_SPECIALIST, P.MAP_SPECIALIST, avg, P.MAP_WEAK, P.MAP_WEAK)),
                 team(avg)),
        scenario("strong_overall_bad_map", "map",
                 "Globally stronger but substantially worse on this map",
                 team(strong.with_(map_win_rate=0.32, map_games=45,
                                   map_agent_win_rate=0.32, name="strong_bad_map")),
                 team(avg)),
        scenario("weak_overall_good_map", "map",
                 "Globally weaker but much better on this map",
                 team(weak.with_(map_win_rate=0.70, map_games=45,
                                 map_agent_win_rate=0.70, name="weak_good_map")),
                 team(avg)),
    ]

    # --- 7. agent ------------------------------------------------------
    s += [
        scenario("agent_comfort", "agent", "Everyone on an agent they know well",
                 team(P.AGENT_SPECIALIST), team(avg), expect="a_favoured"),
        scenario("agent_unfamiliar", "agent", "Everyone on an agent they are bad with",
                 team(P.AGENT_WEAK), team(avg), expect="b_favoured"),
        scenario("single_agent_specialist", "agent", "One agent specialist",
                 team(*swap(avg, 2, P.AGENT_SPECIALIST)), team(avg)),
        scenario("single_agent_weak", "agent", "One player on a poor agent",
                 team(*swap(avg, 2, P.AGENT_WEAK)), team(avg)),
        scenario("several_agent_specialists", "agent", "Three agent specialists",
                 team(*(P.AGENT_SPECIALIST, P.AGENT_SPECIALIST,
                        P.AGENT_SPECIALIST, avg, avg)), team(avg)),
        scenario("several_weak_agents", "agent", "Three players on poor agents",
                 team(*(P.AGENT_WEAK, P.AGENT_WEAK, P.AGENT_WEAK, avg, avg)),
                 team(avg)),
        scenario("strong_player_bad_agent", "agent",
                 "A strong player on an agent they perform badly with",
                 team(*swap(avg, 2, strong.with_(agent_win_rate=0.32,
                                                 agent_games=90, games=180,
                                                 name="strong_bad_agent"))),
                 team(avg)),
        scenario("weak_player_best_agent", "agent",
                 "A weak player on their single best agent",
                 team(*swap(avg, 2, weak.with_(agent_win_rate=0.70,
                                               agent_games=90, games=180,
                                               name="weak_best_agent"))),
                 team(avg)),
    ]

    # --- 8. map x agent interaction ------------------------------------
    s += [
        scenario("map_agent_specialist", "map_agent",
                 "Ordinary overall, ordinary on map and agent -- excellent combined",
                 team(P.MAP_AGENT_SPECIALIST), team(avg), expect="a_favoured"),
        scenario("map_agent_weak", "map_agent",
                 "Ordinary in isolation, poor in combination",
                 team(P.MAP_AGENT_WEAK), team(avg), expect="b_favoured"),
        scenario("good_map_bad_agent", "map_agent", "Strong map history, weak agent",
                 team(avg.with_(map_win_rate=0.68, map_games=45,
                                agent_win_rate=0.34, agent_games=90, games=180,
                                map_agent_games=20, map_agent_win_rate=0.50,
                                name="good_map_bad_agent")), team(avg)),
        scenario("bad_map_good_agent", "map_agent", "Weak map history, strong agent",
                 team(avg.with_(map_win_rate=0.32, map_games=45,
                                agent_win_rate=0.66, agent_games=90, games=180,
                                map_agent_games=20, map_agent_win_rate=0.50,
                                name="bad_map_good_agent")), team(avg)),
        scenario("map_agent_sparse", "map_agent",
                 "An extreme combined rate on only two games -- shrinkage should bite",
                 team(avg.with_(map_agent_games=2, map_agent_win_rate=1.0,
                                name="sparse_combo")), team(avg)),
    ]

    # --- 9. sample size / shrinkage ------------------------------------
    s += [
        scenario("shrinkage_3_at_100", "shrinkage",
                 "Three games at 100% against a large stable history",
                 team(P.LOW_SAMPLE_HIGH_WR), team(P.BIG_SAMPLE_GOOD_WR),
                 expect="b_favoured"),
        scenario("shrinkage_3_at_0", "shrinkage",
                 "Three games at 0% -- must not read as a hopeless player",
                 team(P.LOW_SAMPLE_LOW_WR), team(avg)),
        scenario("shrinkage_20_at_70", "shrinkage", "Twenty games at 70%",
                 team(avg.with_(games=20, map_games=4, agent_games=7,
                                map_agent_games=2, win_rate=0.70,
                                map_win_rate=0.70, agent_win_rate=0.70,
                                map_agent_win_rate=0.70, name="wr70_n20")),
                 team(avg)),
        scenario("shrinkage_100_at_60", "shrinkage", "A hundred games at 60%",
                 team(avg.with_(games=100, map_games=20, agent_games=40,
                                map_agent_games=10, win_rate=0.60,
                                map_win_rate=0.60, agent_win_rate=0.60,
                                map_agent_win_rate=0.60, name="wr60_n100")),
                 team(avg)),
        scenario("shrinkage_500_at_55", "shrinkage", "Five hundred games at 55%",
                 team(P.BIG_SAMPLE_GOOD_WR), team(avg)),
        scenario("shrinkage_zero_games", "shrinkage", "No history whatsoever",
                 team(P.UNKNOWN), team(avg)),
        scenario("shrinkage_one_game", "shrinkage", "A single prior match, won",
                 team(avg.with_(games=1, map_games=0, agent_games=1,
                                map_agent_games=0, win_rate=1.0,
                                agent_win_rate=1.0, name="one_game_won")),
                 team(avg)),
        scenario("shrinkage_map_sparse", "shrinkage",
                 "Extreme map rate on two games",
                 team(avg.with_(map_games=2, map_win_rate=1.0,
                                map_agent_games=1, map_agent_win_rate=1.0,
                                name="map_sparse")), team(avg)),
        scenario("shrinkage_agent_sparse", "shrinkage",
                 "Extreme agent rate on two games",
                 team(avg.with_(agent_games=2, agent_win_rate=1.0,
                                map_agent_games=1, map_agent_win_rate=1.0,
                                name="agent_sparse")), team(avg)),
        scenario("shrinkage_recent_sparse", "shrinkage",
                 "A perfect recent window over very few games",
                 team(avg.with_(games=4, map_games=1, agent_games=2,
                                map_agent_games=1, recent_win_rate=1.0,
                                name="recent_sparse")), team(avg)),
    ]

    # --- 10. recent form -----------------------------------------------
    s += [
        scenario("hot_streak", "form", "Team A on a hot streak",
                 team(P.HOT), team(avg), expect="a_favoured"),
        scenario("cold_streak", "form", "Team A on a cold streak",
                 team(P.COLD), team(avg), expect="b_favoured"),
        scenario("hot_vs_cold_form", "form", "Hot streak against cold streak",
                 team(P.HOT), team(P.COLD), expect="a_favoured"),
        scenario("mild_improvement", "form", "A gently rising trend",
                 team(avg.with_(trend=0.15, recent_win_rate=0.58,
                                name="mild_up")), team(avg)),
        scenario("strong_improvement", "form", "A steep rising trend",
                 team(avg.with_(trend=0.5, recent_win_rate=0.72,
                                name="steep_up")), team(avg)),
        scenario("mild_decline", "form", "A gently falling trend",
                 team(avg.with_(trend=-0.15, recent_win_rate=0.42,
                                name="mild_down")), team(avg)),
        scenario("strong_decline", "form", "A steep falling trend",
                 team(avg.with_(trend=-0.5, recent_win_rate=0.30,
                                name="steep_down")), team(avg)),
        scenario("great_career_bad_form", "form",
                 "Excellent lifetime record, poor recent window",
                 team(strong.with_(recent_win_rate=0.25, trend=-0.4,
                                   name="career_good_form_bad")), team(avg)),
        scenario("poor_career_great_form", "form",
                 "Mediocre lifetime record, excellent recent window",
                 team(below.with_(recent_win_rate=0.80, trend=0.4,
                                  name="career_bad_form_good")), team(avg)),
    ]

    # --- 11. rust / inactivity ------------------------------------------
    for label, days in (("today", 0.0), ("one_day", 1.0), ("few_days", 4.0),
                        ("one_week", 7.0), ("few_weeks", 21.0),
                        ("very_stale", 120.0)):
        s.append(scenario(f"rust_{label}", "rust",
                          f"Team A last played {days:g} days ago",
                          team(avg.with_(days_since_last=days,
                                         name=f"idle_{label}")), team(avg)))
    s += [
        scenario("elite_but_rusty", "rust", "Elite players, long inactive",
                 team(elite.with_(days_since_last=90.0, name="elite_rusty")),
                 team(avg)),
        scenario("average_but_active", "rust", "Average players, playing daily",
                 team(avg.with_(days_since_last=0.0, name="avg_active")),
                 team(avg.with_(days_since_last=60.0, name="avg_stale"))),
    ]

    # --- 12. experience -------------------------------------------------
    s += [
        scenario("brand_new_accounts", "experience", "Five new accounts",
                 team(P.NEW_PLAYER), team(avg)),
        scenario("low_games", "experience", "Short histories against typical ones",
                 team(avg.with_(games=12, map_games=2, agent_games=4,
                                map_agent_games=1, name="few_games")), team(avg)),
        scenario("very_experienced", "experience", "Very long histories",
                 team(P.EXPERIENCED_AVERAGE), team(avg)),
        scenario("high_level_mediocre", "experience",
                 "High account level, mediocre skill",
                 team(P.HIGH_LEVEL_MEDIOCRE), team(avg)),
        scenario("low_level_strong", "experience",
                 "Low account level, strong skill",
                 team(P.LOW_LEVEL_STRONG), team(avg)),
    ]

    # --- 14. composition ------------------------------------------------
    comps = {
        "balanced": BALANCED,
        "no_controller": ("Jett", "Raze", "Sova", "Killjoy", "Phoenix"),
        "no_sentinel": ("Jett", "Omen", "Sova", "Breach", "Raze"),
        "no_initiator": ("Jett", "Omen", "Killjoy", "Sage", "Raze"),
        "no_duelist": ("Omen", "Brimstone", "Sova", "Killjoy", "Sage"),
        "double_duelist": ("Jett", "Raze", "Omen", "Sova", "Killjoy"),
        "triple_duelist": ("Jett", "Raze", "Phoenix", "Omen", "Killjoy"),
        "quad_duelist": ("Jett", "Raze", "Phoenix", "Reyna", "Omen"),
        "all_controllers": ("Omen", "Brimstone", "Astra", "Viper", "Harbor"),
    }
    for label, agents in comps.items():
        if label == "balanced":
            continue
        s.append(scenario(f"comp_{label}", "composition",
                          f"Team A runs a {label.replace('_', ' ')} composition",
                          team(avg, agents=agents), team(avg, agents=BALANCED),
                          tags=("observational",)))
    s.append(scenario("comp_balanced_vs_skill", "composition",
                      "Balanced composition against a stronger but stacked team",
                      team(avg, agents=BALANCED),
                      team(above, agents=comps["triple_duelist"]),
                      tags=("observational",)))

    # --- 15. off-role ---------------------------------------------------
    duelists = ("Jett", "Raze", "Phoenix", "Reyna", "Neon")
    sentinel_mains = tuple(P.OFF_ROLE for _ in range(5))
    s += [
        scenario("off_role_none", "off_role", "Nobody playing outside their role",
                 team(avg), team(avg), expect="even"),
        scenario("off_role_one", "off_role", "One player outside their usual role",
                 team(*swap(avg, 2, P.OFF_ROLE)), team(avg)),
        scenario("off_role_several", "off_role", "Three players outside their roles",
                 team(*(P.OFF_ROLE, P.OFF_ROLE, P.OFF_ROLE, avg, avg)), team(avg)),
        scenario("off_role_all", "off_role", "Everyone forced off-role",
                 TeamProfile(players=sentinel_mains, agents=duelists), team(avg)),
        scenario("off_role_strong_team", "off_role",
                 "A stronger team playing off-role against a weaker comfort team",
                 TeamProfile(players=tuple(strong.with_(role="Sentinel") for _ in range(5)),
                             agents=duelists),
                 team(below)),
    ]

    # --- 16. party structure --------------------------------------------
    def parties(*sizes):
        """Assign party ids to a five-player roster from group sizes."""
        out, idx = [], 0
        for gi, size in enumerate(sizes):
            for _ in range(size):
                out.append(f"p{gi}" if size > 1 else None)
                idx += 1
        return out

    party_shapes = {
        "five_solos": (1, 1, 1, 1, 1),
        "one_duo": (2, 1, 1, 1),
        "two_duos": (2, 2, 1),
        "trio": (3, 1, 1),
        "trio_plus_duo": (3, 2),
        "four_stack": (4, 1),
        "five_stack": (5,),
    }
    for label, shape in party_shapes.items():
        ids = parties(*shape)
        roster = tuple(avg.with_(party=pid, name=f"avg_{label}_{i}")
                       for i, pid in enumerate(ids))
        s.append(scenario(f"party_{label}", "party",
                          f"Team A queued as {label.replace('_', ' ')}",
                          TeamProfile(players=roster, agents=BALANCED),
                          team(avg), tags=("observational",)))
    five_stack = tuple(avg.with_(party="p0", name=f"stack{i}") for i in range(5))
    s.append(scenario("five_stack_vs_solos", "party",
                      "A five-stack of average players against five solo queuers",
                      TeamProfile(players=five_stack, agents=BALANCED),
                      team(avg), tags=("observational",)))
    weak_stack = tuple(below.with_(party="p0", name=f"wstack{i}") for i in range(5))
    s.append(scenario("weak_stack_vs_strong_solos", "party",
                      "A coordinated weaker five-stack against stronger solos",
                      TeamProfile(players=weak_stack, agents=BALANCED),
                      team(above), tags=("observational",)))

    # --- 17. team distribution ------------------------------------------
    def acs_team(values, label):
        return TeamProfile(
            players=tuple(avg.with_(acs=v, adr=P.POP_ADR + (v - P.POP_ACS) * 0.65,
                                    name=f"{label}{i}")
                          for i, v in enumerate(values)),
            agents=BALANCED)

    s += [
        scenario("spread_same_mean_high_max", "distribution",
                 "Equal mean, one far higher peak",
                 acs_team((320, 185, 185, 185, 185), "peak"),
                 acs_team((212, 212, 212, 212, 212), "flat")),
        scenario("spread_same_mean_low_min", "distribution",
                 "Equal mean, one far lower floor",
                 acs_team((250, 250, 250, 250, 100), "floor"),
                 acs_team((220, 220, 220, 220, 220), "flat")),
        scenario("spread_wide_vs_flat", "distribution",
                 "Equal mean, both tails widened",
                 acs_team((320, 260, 212, 160, 108), "wide"),
                 acs_team((212, 212, 212, 212, 212), "flat")),
        scenario("spread_flat_vs_wide", "distribution",
                 "The reverse pairing",
                 acs_team((212, 212, 212, 212, 212), "flat"),
                 acs_team((320, 260, 212, 160, 108), "wide")),
        scenario("spread_two_peaks", "distribution",
                 "Two strong players and three weak, against five average",
                 acs_team((300, 300, 160, 160, 160), "twopeak"),
                 acs_team((216, 216, 216, 216, 216), "flat")),
        scenario("spread_identical", "distribution",
                 "Identical distributions -- the control",
                 acs_team((212, 212, 212, 212, 212), "flat"),
                 acs_team((212, 212, 212, 212, 212), "flat"), expect="even"),
    ]

    # --- 18. coverage / missing history ---------------------------------
    for known in range(0, 6):
        roster = tuple(avg if i < known else P.UNKNOWN for i in range(5))
        s.append(scenario(f"coverage_a_{known}_of_5", "coverage",
                          f"Team A has history for {known} of five players",
                          TeamProfile(players=roster, agents=BALANCED),
                          team(avg)))
    s += [
        scenario("coverage_none_either_side", "coverage",
                 "Neither team has any history at all",
                 team(P.UNKNOWN), team(P.UNKNOWN), expect="even"),
        scenario("coverage_full_vs_none", "coverage",
                 "Complete history against none",
                 team(avg), team(P.UNKNOWN)),
        scenario("coverage_none_vs_full", "coverage",
                 "None against complete history",
                 team(P.UNKNOWN), team(avg)),
        scenario("coverage_2_vs_5", "coverage",
                 "Two known against five known",
                 TeamProfile(players=(avg, avg, P.UNKNOWN, P.UNKNOWN, P.UNKNOWN),
                             agents=BALANCED), team(avg)),
        scenario("coverage_unknown_vs_weak", "coverage",
                 "Unknown players must not be read as terrible ones",
                 team(P.UNKNOWN), team(weak)),
    ]

    # --- 19. contradictory signals ---------------------------------------
    s += [
        scenario("contra_rank_vs_rating", "contradiction",
                 "Higher rank, worse everything else",
                 team(at_tier(below, 21)), team(at_tier(above, 12)),
                 tags=("observational",)),
        scenario("contra_rating_vs_rank", "contradiction",
                 "Lower rank, much stronger rating",
                 team(at_tier(strong, 12)), team(at_tier(avg, 22)),
                 tags=("observational",)),
        scenario("contra_overall_vs_map", "contradiction",
                 "Good overall record, poor map record",
                 team(strong.with_(map_win_rate=0.30, map_games=45,
                                   map_agent_win_rate=0.30, name="good_bad_map")),
                 team(avg), tags=("observational",)),
        scenario("contra_map_vs_overall", "contradiction",
                 "Poor overall record, excellent map record",
                 team(weak.with_(map_win_rate=0.72, map_games=45,
                                 map_agent_win_rate=0.72, name="bad_good_map")),
                 team(avg), tags=("observational",)),
        scenario("contra_map_vs_agent", "contradiction",
                 "Strong on the map, weak on the agent",
                 team(avg.with_(map_win_rate=0.70, map_games=45,
                                agent_win_rate=0.32, agent_games=90, games=200,
                                map_agent_games=20, map_agent_win_rate=0.50,
                                name="map_good_agent_bad")), team(avg),
                 tags=("observational",)),
        scenario("contra_agent_vs_combo", "contradiction",
                 "Strong on the agent, weak in the map-agent combination",
                 team(avg.with_(agent_win_rate=0.68, agent_games=90, games=200,
                                map_games=40, map_agent_games=25,
                                map_agent_win_rate=0.24,
                                name="agent_good_combo_bad")), team(avg),
                 tags=("observational",)),
        scenario("contra_form_vs_career", "contradiction",
                 "Hot form, weak career",
                 team(weak.with_(recent_win_rate=0.80, trend=0.4,
                                 name="cold_career_hot_form")), team(avg),
                 tags=("observational",)),
        scenario("contra_career_vs_form", "contradiction",
                 "Strong career, cold form",
                 team(strong.with_(recent_win_rate=0.22, trend=-0.4,
                                   name="hot_career_cold_form")), team(avg),
                 tags=("observational",)),
        scenario("contra_skill_vs_comp", "contradiction",
                 "Stronger players, worse composition",
                 team(strong, agents=comps["quad_duelist"]),
                 team(avg, agents=BALANCED), tags=("observational",)),
        scenario("contra_comp_vs_skill", "contradiction",
                 "Weaker players, better composition",
                 team(below, agents=BALANCED),
                 team(above, agents=comps["quad_duelist"]),
                 tags=("observational",)),
        scenario("contra_stack_vs_skill", "contradiction",
                 "Coordinated weaker five-stack against stronger solos",
                 TeamProfile(players=weak_stack, agents=BALANCED),
                 team(strong), tags=("observational",)),
        scenario("contra_carry_vs_balance", "contradiction",
                 "One elite carry against a uniformly solid team",
                 team(*swap(weak, 2, elite)), team(above),
                 tags=("observational",)),
    ]

    # --- 20. boundaries --------------------------------------------------
    edges = [
        ("wr_zero", dict(win_rate=0.0, map_win_rate=0.0, agent_win_rate=0.0,
                         map_agent_win_rate=0.0)),
        ("wr_one", dict(win_rate=1.0, map_win_rate=1.0, agent_win_rate=1.0,
                        map_agent_win_rate=1.0)),
        ("kast_zero", dict(kast=0.0)),
        ("kast_one", dict(kast=1.0)),
        ("fb_zero", dict(fb_rate=0.0)),
        ("fd_zero", dict(fd_rate=0.0)),
        ("fb_max", dict(fb_rate=0.30)),
        ("acs_floor", dict(acs=0.0, adr=0.0)),
        ("acs_ceiling", dict(acs=600.0, adr=400.0)),
        ("tier_min", dict(tier=0)),
        ("tier_max", dict(tier=27)),
        ("level_min", dict(account_level=1)),
        ("level_max", dict(account_level=3000)),
        ("idle_zero", dict(days_since_last=0.0)),
        ("idle_max", dict(days_since_last=1500.0)),
    ]
    for label, changes in edges:
        s.append(scenario(f"edge_{label}", "boundary",
                          f"Team A at the {label.replace('_', ' ')} boundary",
                          team(avg.with_(name=f"edge_{label}", **changes)),
                          team(avg), tags=("boundary",)))

    # --- 21. graded dominance --------------------------------------------
    for label, prof, exp in (("slight", above, "a_favoured"),
                             ("moderate", strong, "a_favoured"),
                             ("strong", elite, "a_favoured")):
        s.append(scenario(f"dominance_{label}", "dominance",
                          f"Team A better on every positive factor ({label})",
                          team(prof), team(below if label != "slight" else avg),
                          expect=exp))
    s += [
        scenario("dominance_extreme", "dominance",
                 "Team A better on essentially every measurable factor",
                 team(elite.with_(recent_win_rate=0.78, map_win_rate=0.72,
                                  agent_win_rate=0.72, map_agent_win_rate=0.72,
                                  map_games=45, days_since_last=0.5,
                                  trend=0.35, name="dominant")),
                 team(weak.with_(recent_win_rate=0.22, map_win_rate=0.28,
                                 agent_win_rate=0.28, map_agent_win_rate=0.28,
                                 map_games=45, days_since_last=60.0,
                                 trend=-0.35, name="dominated")),
                 expect="a_favoured"),
        scenario("full_dominance", "dominance",
                 "Alias of the extreme case, kept for the CLI examples",
                 team(elite.with_(recent_win_rate=0.78, trend=0.3,
                                  name="dominant2")),
                 team(weak.with_(recent_win_rate=0.22, trend=-0.3,
                                 name="dominated2")), expect="a_favoured"),
    ]

    # --- 22. cancellation --------------------------------------------------
    s += [
        scenario("cancel_rank_vs_rating", "cancellation",
                 "Team A ranked higher, Team B rates better",
                 team(at_tier(avg, 20)), team(at_tier(above, 14)),
                 tags=("observational",)),
        scenario("cancel_map_vs_agent", "cancellation",
                 "Team A better on the map, Team B better on the agent",
                 team(avg.with_(map_win_rate=0.66, map_games=40, name="a_map")),
                 team(avg.with_(agent_win_rate=0.66, agent_games=90, games=180,
                                name="b_agent")), tags=("observational",)),
        scenario("cancel_form_vs_party", "cancellation",
                 "Team A in better form, Team B queued together",
                 team(P.HOT),
                 TeamProfile(players=five_stack, agents=BALANCED),
                 tags=("observational",)),
        scenario("cancel_carry_vs_floor", "cancellation",
                 "Team A has the higher ceiling, Team B the higher floor",
                 acs_team((320, 180, 180, 180, 180), "ceil"),
                 acs_team((240, 240, 240, 200, 200), "floorhigh"),
                 tags=("observational",)),
    ]

    return s


CATALOG: list[MatchScenario] = build()
BY_NAME: dict[str, MatchScenario] = {s.name: s for s in CATALOG}


def get(name: str) -> MatchScenario:
    if name not in BY_NAME:
        raise KeyError(f"unknown scenario {name!r}")
    return BY_NAME[name]


def categories() -> dict[str, int]:
    out: dict[str, int] = {}
    for s in CATALOG:
        out[s.category] = out.get(s.category, 0) + 1
    return out
