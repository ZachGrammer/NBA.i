"""
structured_answers.py
---------------------
Structured answer helpers for leaderboard, recent-form, shot-zone,
and comparison queries using the CSV outputs from ingest.py.

Structured answers hard-filter by season when provided.
If no season is provided, they default to the latest completed season.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

SEASON_STATS_PATH = os.path.join(PROCESSED_DIR, "season_player_stats.csv")
RECENT_GAMES_PATH = os.path.join(PROCESSED_DIR, "recent_player_games.csv")
SHOT_PROFILE_PATH = os.path.join(PROCESSED_DIR, "player_shot_profile.csv")
SHOT_PROFILE_OVERALL_PATH = os.path.join(PROCESSED_DIR, "player_shot_profile_overall.csv")

LATEST_COMPLETED_SEASON = "2024-25"


# -------------------------------------------------------------------
# load helpers
# -------------------------------------------------------------------


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_season_stats() -> pd.DataFrame:
    return _load_csv(SEASON_STATS_PATH)


def load_recent_games() -> pd.DataFrame:
    df = _load_csv(RECENT_GAMES_PATH)
    if not df.empty and "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def load_shot_profile() -> pd.DataFrame:
    return _load_csv(SHOT_PROFILE_PATH)


def load_shot_profile_overall() -> pd.DataFrame:
    return _load_csv(SHOT_PROFILE_OVERALL_PATH)


# -------------------------------------------------------------------
# general helpers
# -------------------------------------------------------------------


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def _normalize_name(text: str) -> str:
    text = _strip_accents(text.lower())
    return re.sub(r"[^a-z0-9 ]", "", text).strip()


def extract_player_names(query: str, valid_names: list[str], max_players: int = 2) -> list[str]:
    """
    Improved player extraction:
    - matches full names
    - matches last names
    - matches distinctive first names
    - strips accents for names like Jokić -> Jokic
    """
    q = _normalize_name(query)
    matches = []

    for name in valid_names:
        name_norm = _normalize_name(name)
        parts = name_norm.split()

        if name_norm in q:
            matches.append(name)
            continue

        if parts:
            last_name = parts[-1]
            if len(last_name) >= 4 and last_name in q:
                matches.append(name)
                continue

        if parts:
            first_name = parts[0]
            if len(first_name) >= 5 and first_name in q:
                matches.append(name)
                continue

    matches = sorted(set(matches), key=len, reverse=True)
    return matches[:max_players]


def _format_pct(val: float) -> str:
    return f"{float(val):.1f}%"


def _format_float(val: float) -> str:
    return f"{float(val):.1f}"


def _safe_top(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df.head(n).reset_index(drop=True)


def _resolve_season(season: str | None) -> str:
    return season or LATEST_COMPLETED_SEASON


def _filter_by_season(df: pd.DataFrame, season: str | None) -> pd.DataFrame:
    if df.empty or "season" not in df.columns:
        return df
    resolved = _resolve_season(season)
    return df[df["season"] == resolved].copy()


# -------------------------------------------------------------------
# season leaderboard
# -------------------------------------------------------------------


SEASON_STAT_MAP = {
    "points": "points_per_game",
    "rebounds": "rebounds_per_game",
    "assists": "assists_per_game",
    "steals": "steals_per_game",
    "blocks": "blocks_per_game",
    "fg_pct": "fg_pct",
    "fg3_pct": "fg3_pct",
    "ft_pct": "ft_pct",
}


def answer_season_leaderboard(stat: str, top_n: int = 5, season: str | None = None) -> Optional[dict]:
    df = load_season_stats()
    if df.empty:
        return None

    df = _filter_by_season(df, season)
    stat_col = SEASON_STAT_MAP.get(stat)
    if stat_col is None or stat_col not in df.columns:
        return None

    filtered = df.copy()

    if stat == "fg3_pct" and "fg3a_per_game" in filtered.columns:
        filtered = filtered[filtered["fg3a_per_game"] >= 2.0]
    if stat == "fg_pct" and "fga_per_game" in filtered.columns:
        filtered = filtered[filtered["fga_per_game"] >= 5.0]
    if stat == "ft_pct" and "fta_per_game" in filtered.columns:
        filtered = filtered[filtered["fta_per_game"] >= 2.0]

    filtered = filtered.sort_values(stat_col, ascending=False)
    filtered = _safe_top(filtered, top_n)

    if filtered.empty:
        return None

    lines = []
    for i, row in filtered.iterrows():
        value = row[stat_col]
        if stat in {"fg_pct", "fg3_pct", "ft_pct"}:
            value_text = _format_pct(value)
        else:
            value_text = _format_float(value)

        lines.append(
            f"{i+1}. {row['player_name']} ({row['team_abbreviation']}) — {value_text}"
        )

    answer = (
        f"Top {top_n} players for {stat.replace('_', ' ')} in {_resolve_season(season)}:\n\n"
        + "\n".join(lines)
    )

    return {
        "answer": answer,
        "mode": "structured_season_leaderboard",
        "rows": filtered.to_dict(orient="records"),
    }


# -------------------------------------------------------------------
# recent leaderboard (last 5 games)
# -------------------------------------------------------------------


RECENT_STAT_MAP = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "steals": "steals",
    "blocks": "blocks",
}


def answer_recent_leaderboard(
    stat: str,
    top_n: int = 5,
    games_n: int = 5,
    use_average: bool = False,
    season: str | None = None,
) -> Optional[dict]:
    df = load_recent_games()
    if df.empty:
        return None

    df = _filter_by_season(df, season)
    stat_col = RECENT_STAT_MAP.get(stat)
    if stat_col is None or stat_col not in df.columns:
        return None

    df = df.sort_values(["player_name", "game_date"], ascending=[True, False]).copy()
    df = df.groupby("player_name", group_keys=False).head(games_n).copy()

    grouped = (
        df.groupby(["player_name", "team_abbreviation", "season"], as_index=False)
        .agg(
            games_used=("game_id", "count"),
            stat_value=(stat_col, "mean" if use_average else "sum"),
        )
    )

    grouped = grouped[grouped["games_used"] >= min(3, games_n)]
    grouped = grouped.sort_values("stat_value", ascending=False)
    grouped = _safe_top(grouped, top_n)

    if grouped.empty:
        return None

    stat_label = f"average {stat}" if use_average else f"total {stat}"
    lines = []
    for i, row in grouped.iterrows():
        lines.append(
            f"{i+1}. {row['player_name']} ({row['team_abbreviation']}) — "
            f"{_format_float(row['stat_value'])} {stat_label} over {int(row['games_used'])} recent games"
        )

    answer = (
        f"Top {top_n} players by {stat_label} in their last {games_n} games of {_resolve_season(season)}:\n\n"
        + "\n".join(lines)
    )

    return {
        "answer": answer,
        "mode": "structured_recent_leaderboard",
        "rows": grouped.to_dict(orient="records"),
    }


# -------------------------------------------------------------------
# player summary / recent summary
# -------------------------------------------------------------------


def answer_player_summary(query: str, season: str | None = None) -> Optional[dict]:
    season_df = load_season_stats()
    recent_df = load_recent_games()
    overall_df = load_shot_profile_overall()

    if season_df.empty:
        return None

    season_df = _filter_by_season(season_df, season)
    recent_df = _filter_by_season(recent_df, season)
    overall_df = _filter_by_season(overall_df, season)

    valid_names = season_df["player_name"].dropna().astype(str).unique().tolist()
    players = extract_player_names(query, valid_names, max_players=1)

    if not players:
        return None

    player = players[0]
    season_row = season_df[season_df["player_name"] == player]
    if season_row.empty:
        return None

    r = season_row.iloc[0]

    lines = [
        f"{player} is averaging {_format_float(r['points_per_game'])} points, "
        f"{_format_float(r['rebounds_per_game'])} rebounds, and "
        f"{_format_float(r['assists_per_game'])} assists per game in {_resolve_season(season)}.",
        f"He is shooting {_format_pct(r['fg_pct'])} from the field and {_format_pct(r['fg3_pct'])} from three."
    ]

    if not recent_df.empty:
        recent_rows = recent_df[recent_df["player_name"] == player].copy()
        if not recent_rows.empty:
            recent_rows = recent_rows.sort_values("game_date", ascending=False).head(5)
            avg_pts = recent_rows["points"].mean()
            avg_reb = recent_rows["rebounds"].mean()
            avg_ast = recent_rows["assists"].mean()
            lines.append(
                f"Over his last {len(recent_rows)} games, he has averaged "
                f"{_format_float(avg_pts)} points, {_format_float(avg_reb)} rebounds, "
                f"and {_format_float(avg_ast)} assists."
            )

    if not overall_df.empty:
        overall_row = overall_df[overall_df["player_name"] == player]
        if not overall_row.empty:
            o = overall_row.iloc[0]
            lines.append(
                f"He takes {_format_float(o['rim_attempt_rate'] * 100)}% of his shots at the rim and "
                f"{_format_float(o['above_break3_attempt_rate'] * 100)}% above the break."
            )

    return {
        "answer": "\n".join(lines),
        "mode": "structured_player_summary",
        "rows": [r.to_dict()],
    }


def answer_player_recent_summary(query: str, games_n: int = 5, season: str | None = None) -> Optional[dict]:
    season_df = load_season_stats()
    recent_df = load_recent_games()

    if season_df.empty or recent_df.empty:
        return None

    season_df = _filter_by_season(season_df, season)
    recent_df = _filter_by_season(recent_df, season)

    valid_names = season_df["player_name"].dropna().astype(str).unique().tolist()
    players = extract_player_names(query, valid_names, max_players=1)

    if not players:
        return None

    player = players[0]
    player_games = recent_df[recent_df["player_name"] == player].copy()

    if player_games.empty:
        return None

    player_games = player_games.sort_values("game_date", ascending=False).head(games_n).copy()
    games_used = len(player_games)

    avg_points = player_games["points"].mean() if "points" in player_games.columns else 0
    avg_rebounds = player_games["rebounds"].mean() if "rebounds" in player_games.columns else 0
    avg_assists = player_games["assists"].mean() if "assists" in player_games.columns else 0
    avg_fg_pct = player_games["fg_pct"].mean() if "fg_pct" in player_games.columns else 0
    avg_fg3_pct = player_games["fg3_pct"].mean() if "fg3_pct" in player_games.columns else 0

    latest_game = player_games.iloc[0]

    lines = [
        f"{player} has averaged {_format_float(avg_points)} points, {_format_float(avg_rebounds)} rebounds, and "
        f"{_format_float(avg_assists)} assists over his last {games_used} games in {_resolve_season(season)}.",
        f"He has shot {_format_pct(avg_fg_pct)} from the field and {_format_pct(avg_fg3_pct)} from three over that span.",
        "",
        f"Most recent game ({latest_game.get('game_date', 'unknown date')} vs {latest_game.get('matchup', 'unknown matchup')}):",
        f"- {_format_float(latest_game.get('points', 0))} points",
        f"- {_format_float(latest_game.get('rebounds', 0))} rebounds",
        f"- {_format_float(latest_game.get('assists', 0))} assists",
        f"- {_format_pct(latest_game.get('fg_pct', 0))} FG",
        f"- {_format_pct(latest_game.get('fg3_pct', 0))} 3PT",
    ]

    return {
        "answer": "\n".join(lines),
        "mode": "structured_player_recent_summary",
        "rows": player_games.to_dict(orient="records"),
    }


# -------------------------------------------------------------------
# shot zone leaderboard
# -------------------------------------------------------------------


def _aggregate_corner_three(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["split_type"] == "shot_zone_basic"].copy()
    df = df[df["split_value"].isin(["Left Corner 3", "Right Corner 3"])].copy()

    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby(["player_id", "player_name", "team_abbreviation", "season"], as_index=False)
        .agg(
            attempts=("attempts", "sum"),
            makes=("makes", "sum"),
        )
    )
    grouped["fg_pct"] = (grouped["makes"] / grouped["attempts"] * 100).fillna(0).round(1)
    grouped["split_value"] = "Corner 3"
    return grouped


def answer_shot_zone_leaderboard(
    zone: Optional[str],
    top_n: int = 5,
    min_attempts: int = 25,
    season: str | None = None,
) -> Optional[dict]:
    df = load_shot_profile()
    if df.empty:
        return None

    df = _filter_by_season(df, season)

    if zone == "corner3_combined":
        grouped = _aggregate_corner_three(df)
    else:
        grouped = df[df["split_type"] == "shot_zone_basic"].copy()
        if zone:
            grouped = grouped[grouped["split_value"] == zone].copy()

    if grouped.empty:
        return None

    grouped = grouped[grouped["attempts"] >= min_attempts].copy()
    if grouped.empty:
        return None

    grouped = grouped.sort_values(["fg_pct", "attempts"], ascending=[False, False])
    grouped = _safe_top(grouped, top_n)

    best = grouped.iloc[0]
    zone_label = best["split_value"]

    lines = [
        f"{best['player_name']} ({best['team_abbreviation']}) is the most efficient shooter from {zone_label.lower()} in {_resolve_season(season)}, "
        f"shooting {_format_pct(best['fg_pct'])} on {int(best['attempts'])} attempts.",
        "",
        f"Other top shooters from {zone_label.lower()} (minimum {min_attempts} attempts):",
    ]

    for _, row in grouped.iloc[1:].iterrows():
        lines.append(
            f"- {row['player_name']} ({row['team_abbreviation']}): {_format_pct(row['fg_pct'])} on {int(row['attempts'])} attempts"
        )

    return {
        "answer": "\n".join(lines),
        "mode": "structured_shot_zone_leaderboard",
        "rows": grouped.to_dict(orient="records"),
    }


# -------------------------------------------------------------------
# shot zone lookup / player shot profile
# -------------------------------------------------------------------


def answer_shot_zone_lookup(query: str, zone: Optional[str], season: str | None = None) -> Optional[dict]:
    shot_df = load_shot_profile()
    overall_df = load_shot_profile_overall()
    season_df = load_season_stats()

    if shot_df.empty or season_df.empty:
        return None

    shot_df = _filter_by_season(shot_df, season)
    overall_df = _filter_by_season(overall_df, season)
    season_df = _filter_by_season(season_df, season)

    valid_names = season_df["player_name"].dropna().astype(str).unique().tolist()
    players = extract_player_names(query, valid_names, max_players=1)
    if not players:
        return None

    player = players[0]
    player_rows = shot_df[shot_df["player_name"] == player].copy()
    overall_row = overall_df[overall_df["player_name"] == player].copy() if not overall_df.empty else pd.DataFrame()

    if player_rows.empty:
        return None

    lines = [f"{player}'s shot profile in {_resolve_season(season)}:"]

    if zone:
        zone_rows = player_rows[
            (player_rows["split_type"] == "shot_zone_basic") &
            (
                (player_rows["split_value"] == zone) |
                (
                    zone == "corner3_combined" and
                    player_rows["split_value"].isin(["Left Corner 3", "Right Corner 3"])
                )
            )
        ].copy()

        if zone == "corner3_combined" and not zone_rows.empty:
            attempts = zone_rows["attempts"].sum()
            makes = zone_rows["makes"].sum()
            fg_pct = (makes / attempts * 100) if attempts else 0
            lines.append(f"- Corner 3: {_format_pct(fg_pct)} on {int(attempts)} attempts")
        elif not zone_rows.empty:
            for _, row in zone_rows.iterrows():
                lines.append(
                    f"- {row['split_value']}: {_format_pct(row['fg_pct'])} on {int(row['attempts'])} attempts"
                )
    else:
        basic_rows = player_rows[player_rows["split_type"] == "shot_zone_basic"].copy()
        basic_rows = basic_rows.sort_values("attempts", ascending=False).head(6)
        for _, row in basic_rows.iterrows():
            lines.append(
                f"- {row['split_value']}: {_format_pct(row['fg_pct'])} on {int(row['attempts'])} attempts"
            )

    if not overall_row.empty:
        r = overall_row.iloc[0]
        lines.append("")
        lines.append("Overall shot mix:")
        lines.append(f"- Rim attempt rate: {_format_float(r.get('rim_attempt_rate', 0) * 100)}%")
        lines.append(f"- Mid-range attempt rate: {_format_float(r.get('midrange_attempt_rate', 0) * 100)}%")
        lines.append(f"- Corner 3 attempt rate: {_format_float(r.get('corner3_attempt_rate', 0) * 100)}%")
        lines.append(f"- Above-break 3 attempt rate: {_format_float(r.get('above_break3_attempt_rate', 0) * 100)}%")

    return {
        "answer": "\n".join(lines),
        "mode": "structured_shot_zone_lookup",
        "rows": player_rows.to_dict(orient="records"),
    }


# -------------------------------------------------------------------
# player comparison
# -------------------------------------------------------------------


def answer_player_comparison(query: str, season: str | None = None) -> Optional[dict]:
    season_df = load_season_stats()
    overall_df = load_shot_profile_overall()

    if season_df.empty:
        return None

    season_df = _filter_by_season(season_df, season)
    overall_df = _filter_by_season(overall_df, season)

    valid_names = season_df["player_name"].dropna().astype(str).unique().tolist()
    players = extract_player_names(query, valid_names, max_players=2)

    if len(players) < 2:
        print(f"[COMPARISON DEBUG] Could not resolve two players from query: {query}")
        print(f"[COMPARISON DEBUG] Matched players: {players}")
        return None

    p1, p2 = players[0], players[1]
    row1 = season_df[season_df["player_name"] == p1]
    row2 = season_df[season_df["player_name"] == p2]

    if row1.empty or row2.empty:
        return None

    r1 = row1.iloc[0]
    r2 = row2.iloc[0]

    lines = [
        f"Comparison: {p1} vs {p2} in {_resolve_season(season)}",
        "",
        f"Points per game: {p1} {_format_float(r1['points_per_game'])} | {p2} {_format_float(r2['points_per_game'])}",
        f"Rebounds per game: {p1} {_format_float(r1['rebounds_per_game'])} | {p2} {_format_float(r2['rebounds_per_game'])}",
        f"Assists per game: {p1} {_format_float(r1['assists_per_game'])} | {p2} {_format_float(r2['assists_per_game'])}",
        f"FG%: {p1} {_format_pct(r1['fg_pct'])} | {p2} {_format_pct(r2['fg_pct'])}",
        f"3P%: {p1} {_format_pct(r1['fg3_pct'])} | {p2} {_format_pct(r2['fg3_pct'])}",
    ]

    if not overall_df.empty:
        o1 = overall_df[overall_df["player_name"] == p1]
        o2 = overall_df[overall_df["player_name"] == p2]
        if not o1.empty and not o2.empty:
            a = o1.iloc[0]
            b = o2.iloc[0]
            lines.extend([
                "",
                "Shot profile:",
                f"Rim attempt rate: {p1} {_format_float(a['rim_attempt_rate'] * 100)}% | {p2} {_format_float(b['rim_attempt_rate'] * 100)}%",
                f"Corner 3 attempt rate: {p1} {_format_float(a['corner3_attempt_rate'] * 100)}% | {p2} {_format_float(b['corner3_attempt_rate'] * 100)}%",
                f"Above-break 3 attempt rate: {p1} {_format_float(a['above_break3_attempt_rate'] * 100)}% | {p2} {_format_float(b['above_break3_attempt_rate'] * 100)}%",
            ])

    return {
        "answer": "\n".join(lines),
        "mode": "structured_player_comparison",
        "rows": [r1.to_dict(), r2.to_dict()],
    }


# -------------------------------------------------------------------
# master dispatcher
# -------------------------------------------------------------------


def answer_structured_query(route_info: dict, query: str) -> Optional[dict]:
    intent = route_info.get("intent")
    stat = route_info.get("stat")
    zone = route_info.get("zone")
    season = route_info.get("season")

    if intent == "season_leaderboard":
        return answer_season_leaderboard(stat=stat, top_n=5, season=season)

    if intent == "recent_leaderboard":
        use_average = "average" in query.lower() or "avg" in query.lower()
        return answer_recent_leaderboard(stat=stat, top_n=5, games_n=5, use_average=use_average, season=season)

    if intent == "player_recent_summary":
        return answer_player_recent_summary(query=query, games_n=5, season=season)

    if intent == "player_summary":
        return answer_player_summary(query=query, season=season)

    if intent == "shot_zone_leaderboard":
        min_attempts = 20 if zone == "corner3_combined" else 25
        return answer_shot_zone_leaderboard(zone=zone, top_n=5, min_attempts=min_attempts, season=season)

    if intent == "shot_zone_lookup":
        return answer_shot_zone_lookup(query=query, zone=zone, season=season)

    if intent == "player_comparison":
        return answer_player_comparison(query=query, season=season)

    return None