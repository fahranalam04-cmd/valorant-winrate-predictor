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
