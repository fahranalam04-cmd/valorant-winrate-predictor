"""Career totals: the counting stats the scoreboard prints.

Two things matter here. The aggregate is a *new* path into `match_players`, so
it has to obey the same `started_at < as_of` firewall as every other read in
`store/temporal.py` -- a display query that quietly included the current match
would show the player their own scoreline as prior history.

And the arithmetic has to be right, because nothing downstream would notice if
it were not: a wrong headshot percentage is still a plausible-looking number.
"""

from __future__ import annotations

from valwr.store import schema, temporal

NOW = 2_000_000_000


def _db(tmp_path, rows):
    """`rows` of (started_at, kills, deaths, assists, hs, bs, ls, score,
    rounds, won)."""
    conn = schema.connect(tmp_path / "c.db")
    schema.create_all(conn)
    for i, r in enumerate(rows):
        ts, k, d, a, hs, bs, ls, sc, rp, won = r
        mid = f"m{i}"
        conn.execute(
            "INSERT INTO matches (match_id, started_at, map, mode, queue, "
            "region, season, rounds_red, rounds_blue, winner, data_quality, "
            "ingested_at) VALUES (?, ?, 'Ascent', 'competitive', 'Standard', "
            "'na', 's', 9, 13, 'Blue', NULL, 0)", (mid, ts))
        conn.execute(
            "INSERT INTO match_players (match_id, puuid, team, agent, "
            "party_id, tier, account_level, score, kills, deaths, assists, "
            "headshots, bodyshots, legshots, damage_dealt, damage_taken, "
            "started_at, map, won, rounds_played) VALUES "
            "(?, 'p', 'Blue', 'Jett', NULL, 15, 100, ?, ?, ?, ?, ?, ?, ?, "
            "3000, 3000, ?, 'Ascent', ?, ?)",
            (mid, sc, k, d, a, hs, bs, ls, ts, won, rp))
    conn.commit()
    return conn


ONE = (NOW - 3600, 15, 10, 5, 30, 60, 10, 4000, 20, 1)
TWO = (NOW - 7200, 5, 20, 3, 10, 80, 10, 2000, 20, 0)


def test_totals_add_up(tmp_path):
    c = temporal.career_totals(_db(tmp_path, [ONE, TWO]), "p", NOW)
    assert (c.games, c.kills, c.deaths, c.assists) == (2, 20, 30, 8)
    assert (c.score, c.rounds) == (6000, 40)
    assert (c.wins, c.decided) == (1, 2)


def test_the_derived_numbers_are_the_ones_a_scoreboard_shows(tmp_path):
    c = temporal.career_totals(_db(tmp_path, [ONE, TWO]), "p", NOW)
    assert c.acs == 6000 / 40 == 150.0
    assert c.kd == 20 / 30
    # Headshots over every shot that landed -- 40 of 200 -- not over kills.
    assert c.headshot_rate == 40 / 200 == 0.2
    assert c.win_rate == 0.5


def test_a_match_starting_exactly_at_as_of_is_excluded(tmp_path):
    """The firewall is `<`, never `<=`. A match must not inform a view of
    itself: at the loading screen this player has two games, not three."""
    now_row = (NOW, 99, 0, 0, 99, 0, 0, 9999, 24, 1)
    conn = _db(tmp_path, [ONE, TWO, now_row])
    assert temporal.career_totals(conn, "p", NOW).games == 2
    assert temporal.career_totals(conn, "p", NOW + 1).games == 3


def test_no_history_is_zero_games_and_no_opinion(tmp_path):
    c = temporal.career_totals(_db(tmp_path, []), "nobody", NOW)
    assert c.games == 0
    assert c.acs is None and c.kd is None
    assert c.headshot_rate is None and c.win_rate is None


def test_a_player_who_never_died_is_not_a_division_by_zero(tmp_path):
    flawless = (NOW - 60, 12, 0, 4, 20, 40, 0, 5000, 20, 1)
    c = temporal.career_totals(_db(tmp_path, [flawless]), "p", NOW)
    assert c.kd == 12.0


def test_from_rows_agrees_with_the_aggregate(tmp_path):
    """The per-map block tallies rows already in memory; the career block runs
    SQL. Two code paths for one definition, so they are checked against each
    other rather than trusted."""
    conn = _db(tmp_path, [ONE, TWO])
    sql = temporal.career_totals(conn, "p", NOW)
    rows = temporal.player_history(conn, "p", NOW)
    mem = temporal.Career.from_rows(rows)
    assert (mem.games, mem.kills, mem.deaths, mem.assists) == \
           (sql.games, sql.kills, sql.deaths, sql.assists)
    assert (mem.acs, mem.kd, mem.headshot_rate, mem.win_rate) == \
           (sql.acs, sql.kd, sql.headshot_rate, sql.win_rate)


def test_undecided_matches_count_as_games_but_not_toward_win_rate(tmp_path):
    """A draw or an aborted match has `won` NULL. It happened -- it belongs in
    the game count and the ACS -- but it cannot be a loss."""
    unresolved = (NOW - 100, 10, 10, 2, 20, 60, 20, 3000, 20, None)
    c = temporal.career_totals(_db(tmp_path, [ONE, unresolved]), "p", NOW)
    assert c.games == 2
    assert c.decided == 1 and c.wins == 1
    assert c.win_rate == 1.0


# --- recent form -------------------------------------------------------

def test_recent_takes_the_newest_matches_not_the_oldest(tmp_path):
    """The whole point of a form window. Ordering the wrong way round would
    report a player's oldest games as their current form, and every number
    would still look entirely plausible."""
    old_bad = [(NOW - 86400 * (10 - i), 2, 20, 1, 5, 90, 5, 1000, 20, 0)
               for i in range(5)]                     # ancient, terrible
    new_good = [(NOW - 3600 * (5 - i), 25, 5, 8, 50, 45, 5, 6000, 20, 1)
                for i in range(5)]                    # recent, excellent
    conn = _db(tmp_path, old_bad + new_good)

    recent = temporal.recent_totals(conn, "p", NOW, last_n=5)
    assert recent.games == 5
    assert recent.kills == 125 and recent.deaths == 25, "took the wrong five"
    assert recent.win_rate == 1.0
    # and the career figure still spans everything
    assert temporal.career_totals(conn, "p", NOW).games == 10


def test_recent_is_capped_by_what_exists(tmp_path):
    """Most players here have fewer than 20 stored matches -- the median is 8 --
    so the window is usually the whole history."""
    c = temporal.recent_totals(_db(tmp_path, [ONE, TWO]), "p", NOW, last_n=20)
    assert c.games == 2


def test_recent_obeys_the_same_time_firewall(tmp_path):
    conn = _db(tmp_path, [ONE, TWO, (NOW, 99, 0, 0, 9, 0, 0, 9999, 24, 1)])
    assert temporal.recent_totals(conn, "p", NOW, last_n=20).games == 2


def test_kd_is_pooled_not_an_average_of_ratios(tmp_path):
    """Two defensible definitions, and they disagree by enough to matter.

    Pooled is total kills over total deaths. The alternative -- averaging each
    match's own K/D -- is what most third-party trackers show, and it reads
    higher because a game with very few deaths produces a huge ratio that an
    average lets dominate. On one real account the same twenty games gave
    0.894 pooled against 0.921 averaged.

    Pooled is the right aggregate here: it weights a long game more than a
    short one, which is what "how does this player do" should mean.
    """
    # 19 ordinary games and one 10-kill, 1-death outlier.
    rows = [(NOW - 3600 * (i + 2), 10, 10, 2, 20, 60, 20, 3000, 20, 1)
            for i in range(19)]
    rows.append((NOW - 3600, 10, 1, 2, 20, 60, 20, 3000, 20, 1))
    c = temporal.recent_totals(_db(tmp_path, rows), "p", NOW, last_n=20)

    pooled = c.kd
    averaged = (19 * (10 / 10) + (10 / 1)) / 20
    assert pooled == 200 / 191
    assert abs(pooled - 1.047) < 0.001
    assert abs(averaged - 1.45) < 0.001
    assert pooled < averaged, "the outlier must not dominate a pooled figure"
