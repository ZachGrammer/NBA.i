"""
ingest.py
---------
Multi-season structured NBA ingest pipeline for the local RAG system.

What this file does:
- Pulls season-level player stats for multiple completed seasons
- Pulls recent game logs for top-N scorers for each season
- Pulls raw shot chart data for selected players for each season
- Aggregates shot data by zone and distance band
- Writes structured CSVs for downstream filtering / analytics
- Writes higher-quality semantic documents with metadata for FAISS

Default seasons:
- 2024-25
- 2023-24
- 2022-23
- 2021-22
- 2020-21

Intentionally excludes 2025-26 because it is incomplete.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    PlayerGameLog,
    ShotChartDetail,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "processed_docs")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_SEASONS = [
    "2024-25",
    "2023-24",
    "2022-23",
    "2021-22",
    "2020-21",
]
LATEST_COMPLETED_SEASON = "2024-25"

DEFAULT_TOP_N_RECENT = 50
DEFAULT_TOP_N_SHOT_PLAYERS = 60
API_SLEEP_SECONDS = 0.65

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val, decimals: int = 3) -> float:
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(val) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return 0


def _distance_band(shot_distance: float) -> str:
    d = _safe_float(shot_distance, 1)
    if d <= 3:
        return "0-3 ft"
    if d <= 8:
        return "4-8 ft"
    if d <= 14:
        return "9-14 ft"
    if d <= 19:
        return "15-19 ft"
    if d <= 24:
        return "20-24 ft"
    return "25+ ft"


def _standardize_game_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def _write_csv(df: pd.DataFrame, filename: str) -> str:
    out_path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"[ingest] Wrote {len(df):,} rows -> {out_path}")
    return out_path


def _write_json(data, filename: str) -> str:
    out_path = os.path.join(DOCS_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[ingest] Wrote JSON -> {out_path}")
    return out_path


def _per_game(total_series: pd.Series, gp_series: pd.Series) -> pd.Series:
    gp = pd.to_numeric(gp_series, errors="coerce").replace(0, pd.NA)
    totals = pd.to_numeric(total_series, errors="coerce").fillna(0.0)
    return (totals / gp).fillna(0.0)


def _doc(text: str, player_name: str, team_abbreviation: str, season: str, doc_type: str) -> dict:
    return {
        "text": text,
        "metadata": {
            "player_name": player_name,
            "team_abbreviation": team_abbreviation,
            "season": season,
            "doc_type": doc_type,
        },
    }


# ---------------------------------------------------------------------------
# Fetch + normalize season player stats
# ---------------------------------------------------------------------------


def fetch_season_player_stats(season: str) -> pd.DataFrame:
    print(f"[ingest] Fetching season player stats for {season}...")
    dash = LeagueDashPlayerStats(season=season)
    raw = dash.get_data_frames()[0].copy()

    if "GP" in raw.columns:
        raw = raw[raw["GP"].fillna(0) > 0].copy()

    keep_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "GP",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "FGA",
        "FG3A",
        "FGM",
        "FG3M",
        "FTA",
        "FTM",
    ]
    keep_cols = [c for c in keep_cols if c in raw.columns]
    df = raw[keep_cols].copy()

    for col in df.columns:
        if col not in {"PLAYER_NAME", "TEAM_ABBREVIATION"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    out = pd.DataFrame({
        "player_id": df["PLAYER_ID"].astype(int),
        "player_name": df["PLAYER_NAME"].astype(str),
        "team_id": df["TEAM_ID"].astype(int),
        "team_abbreviation": df["TEAM_ABBREVIATION"].astype(str),
        "games_played": df["GP"].astype(int),
        "minutes_total": df["MIN"],
        "points_total": df["PTS"],
        "rebounds_total": df["REB"],
        "assists_total": df["AST"],
        "steals_total": df["STL"],
        "blocks_total": df["BLK"],
        "fgm_total": df["FGM"],
        "fga_total": df["FGA"],
        "fg3m_total": df["FG3M"],
        "fg3a_total": df["FG3A"],
        "ftm_total": df["FTM"],
        "fta_total": df["FTA"],
        "season": season,
    })

    out["minutes_per_game"] = _per_game(df["MIN"], df["GP"]).round(1)
    out["points_per_game"] = _per_game(df["PTS"], df["GP"]).round(1)
    out["rebounds_per_game"] = _per_game(df["REB"], df["GP"]).round(1)
    out["assists_per_game"] = _per_game(df["AST"], df["GP"]).round(1)
    out["steals_per_game"] = _per_game(df["STL"], df["GP"]).round(1)
    out["blocks_per_game"] = _per_game(df["BLK"], df["GP"]).round(1)
    out["fgm_per_game"] = _per_game(df["FGM"], df["GP"]).round(1)
    out["fga_per_game"] = _per_game(df["FGA"], df["GP"]).round(1)
    out["fg3m_per_game"] = _per_game(df["FG3M"], df["GP"]).round(1)
    out["fg3a_per_game"] = _per_game(df["FG3A"], df["GP"]).round(1)
    out["ftm_per_game"] = _per_game(df["FTM"], df["GP"]).round(1)
    out["fta_per_game"] = _per_game(df["FTA"], df["GP"]).round(1)

    out["fg_pct"] = (df["FG_PCT"] * 100).round(1)
    out["fg3_pct"] = (df["FG3_PCT"] * 100).round(1)
    out["ft_pct"] = (df["FT_PCT"] * 100).round(1)

    out = out.sort_values(["points_per_game", "player_name"], ascending=[False, True]).reset_index(drop=True)
    return out


def fetch_all_season_player_stats(seasons: list[str]) -> pd.DataFrame:
    frames = [fetch_season_player_stats(season) for season in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Fetch + normalize recent games
# ---------------------------------------------------------------------------


def fetch_recent_player_games_for_season(
    season_stats_df: pd.DataFrame,
    season: str,
    top_n_players: int = DEFAULT_TOP_N_RECENT,
    games_per_player: int = 5,
) -> pd.DataFrame:
    print(f"[ingest] Fetching recent game logs for top {top_n_players} scorers in {season}...")
    season_df = season_stats_df[season_stats_df["season"] == season].copy()
    leaders_df = season_df.sort_values("points_per_game", ascending=False).head(top_n_players)

    all_rows: list[pd.DataFrame] = []

    for _, row in leaders_df.iterrows():
        player_id = _safe_int(row["player_id"])
        player_name = row["player_name"]
        team_abbreviation = row.get("team_abbreviation", "N/A")

        try:
            time.sleep(API_SLEEP_SECONDS)
            log = PlayerGameLog(player_id=player_id, season=season)
            games_df = log.get_data_frames()[0].copy()

            if games_df.empty:
                continue

            games_df = games_df.head(games_per_player).copy()
            games_df["player_id"] = player_id
            games_df["player_name"] = player_name
            games_df["team_abbreviation"] = team_abbreviation
            games_df["season"] = season

            rename_map = {
                "Game_ID": "game_id",
                "GAME_ID": "game_id",
                "GAME_DATE": "game_date",
                "MATCHUP": "matchup",
                "WL": "win_loss",
                "MIN": "minutes",
                "PTS": "points",
                "REB": "rebounds",
                "AST": "assists",
                "STL": "steals",
                "BLK": "blocks",
                "FGM": "fgm",
                "FGA": "fga",
                "FG_PCT": "fg_pct",
                "FG3M": "fg3m",
                "FG3A": "fg3a",
                "FG3_PCT": "fg3_pct",
                "FTM": "ftm",
                "FTA": "fta",
                "FT_PCT": "ft_pct",
            }
            games_df = games_df.rename(columns=rename_map)

            keep_cols = [
                "player_id",
                "player_name",
                "team_abbreviation",
                "season",
                "game_id",
                "game_date",
                "matchup",
                "win_loss",
                "minutes",
                "points",
                "rebounds",
                "assists",
                "steals",
                "blocks",
                "fgm",
                "fga",
                "fg_pct",
                "fg3m",
                "fg3a",
                "fg3_pct",
                "ftm",
                "fta",
                "ft_pct",
            ]
            keep_cols = [c for c in keep_cols if c in games_df.columns]
            games_df = games_df[keep_cols].copy()

            if "game_date" in games_df.columns:
                games_df["game_date"] = _standardize_game_date(games_df["game_date"])

            for col in [
                "minutes",
                "points",
                "rebounds",
                "assists",
                "steals",
                "blocks",
                "fgm",
                "fga",
                "fg_pct",
                "fg3m",
                "fg3a",
                "fg3_pct",
                "ftm",
                "fta",
                "ft_pct",
            ]:
                if col in games_df.columns:
                    games_df[col] = pd.to_numeric(games_df[col], errors="coerce").fillna(0.0)

            for pct_col in ["fg_pct", "fg3_pct", "ft_pct"]:
                if pct_col in games_df.columns:
                    games_df[pct_col] = (games_df[pct_col] * 100).round(1)

            all_rows.append(games_df)

        except Exception as exc:
            print(f"  [warn] Could not fetch recent games for {player_name} in {season}: {exc}")

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["season", "player_name", "game_date"], ascending=[False, True, False]).reset_index(drop=True)
    return out


def fetch_all_recent_player_games(
    season_stats_df: pd.DataFrame,
    seasons: list[str],
    top_n_players: int = DEFAULT_TOP_N_RECENT,
    games_per_player: int = 5,
) -> pd.DataFrame:
    frames = [
        fetch_recent_player_games_for_season(
            season_stats_df=season_stats_df,
            season=season,
            top_n_players=top_n_players,
            games_per_player=games_per_player,
        )
        for season in seasons
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Fetch + normalize raw shots
# ---------------------------------------------------------------------------


def fetch_raw_shots_for_season(
    season_stats_df: pd.DataFrame,
    season: str,
    top_n_players: int = DEFAULT_TOP_N_SHOT_PLAYERS,
) -> pd.DataFrame:
    print(f"[ingest] Fetching raw shot data for top {top_n_players} scorers in {season}...")
    season_df = season_stats_df[season_stats_df["season"] == season].copy()
    leaders_df = season_df.sort_values("points_per_game", ascending=False).head(top_n_players)

    all_shots: list[pd.DataFrame] = []

    for _, row in leaders_df.iterrows():
        player_id = _safe_int(row["player_id"])
        player_name = row["player_name"]
        team_abbreviation = row.get("team_abbreviation", "N/A")

        try:
            time.sleep(API_SLEEP_SECONDS)
            shot_detail = ShotChartDetail(
                team_id=0,
                player_id=player_id,
                season_nullable=season,
                context_measure_simple="FGA",
            )
            shots_df = shot_detail.get_data_frames()[0].copy()

            if shots_df.empty:
                continue

            rename_map = {
                "GRID_TYPE": "grid_type",
                "GAME_ID": "game_id",
                "GAME_EVENT_ID": "game_event_id",
                "PLAYER_ID": "player_id",
                "PLAYER_NAME": "player_name",
                "TEAM_ID": "team_id",
                "TEAM_NAME": "team_name",
                "PERIOD": "period",
                "MINUTES_REMAINING": "minutes_remaining",
                "SECONDS_REMAINING": "seconds_remaining",
                "EVENT_TYPE": "event_type",
                "ACTION_TYPE": "action_type",
                "SHOT_TYPE": "shot_type",
                "SHOT_ZONE_BASIC": "shot_zone_basic",
                "SHOT_ZONE_AREA": "shot_zone_area",
                "SHOT_ZONE_RANGE": "shot_zone_range",
                "SHOT_DISTANCE": "shot_distance",
                "LOC_X": "loc_x",
                "LOC_Y": "loc_y",
                "SHOT_ATTEMPTED_FLAG": "shot_attempted_flag",
                "SHOT_MADE_FLAG": "shot_made_flag",
                "GAME_DATE": "game_date",
                "HTM": "home_team",
                "VTM": "visitor_team",
            }
            shots_df = shots_df.rename(columns=rename_map)
            shots_df["season"] = season
            shots_df["team_abbreviation"] = team_abbreviation

            keep_cols = [
                "player_id",
                "player_name",
                "team_id",
                "team_name",
                "team_abbreviation",
                "season",
                "game_id",
                "game_event_id",
                "game_date",
                "period",
                "minutes_remaining",
                "seconds_remaining",
                "event_type",
                "action_type",
                "shot_type",
                "shot_zone_basic",
                "shot_zone_area",
                "shot_zone_range",
                "shot_distance",
                "loc_x",
                "loc_y",
                "shot_attempted_flag",
                "shot_made_flag",
                "home_team",
                "visitor_team",
            ]
            keep_cols = [c for c in keep_cols if c in shots_df.columns]
            shots_df = shots_df[keep_cols].copy()

            if "game_date" in shots_df.columns:
                shots_df["game_date"] = _standardize_game_date(shots_df["game_date"])

            numeric_cols = [
                "period",
                "minutes_remaining",
                "seconds_remaining",
                "shot_distance",
                "loc_x",
                "loc_y",
                "shot_attempted_flag",
                "shot_made_flag",
            ]
            for col in numeric_cols:
                if col in shots_df.columns:
                    shots_df[col] = pd.to_numeric(shots_df[col], errors="coerce").fillna(0.0)

            shots_df["distance_band"] = shots_df["shot_distance"].apply(_distance_band)
            all_shots.append(shots_df)

        except Exception as exc:
            print(f"  [warn] Could not fetch shot chart data for {player_name} in {season}: {exc}")

    if not all_shots:
        return pd.DataFrame()

    out = pd.concat(all_shots, ignore_index=True)
    out = out.sort_values(["season", "player_name", "game_date"], ascending=[False, True, False]).reset_index(drop=True)
    return out


def fetch_all_raw_shots(
    season_stats_df: pd.DataFrame,
    seasons: list[str],
    top_n_players: int = DEFAULT_TOP_N_SHOT_PLAYERS,
) -> pd.DataFrame:
    frames = [
        fetch_raw_shots_for_season(
            season_stats_df=season_stats_df,
            season=season,
            top_n_players=top_n_players,
        )
        for season in seasons
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Shot aggregations
# ---------------------------------------------------------------------------


def aggregate_shot_profile(raw_shots_df: pd.DataFrame) -> pd.DataFrame:
    if raw_shots_df.empty:
        return pd.DataFrame()

    base_cols = ["player_id", "player_name", "team_abbreviation", "season"]
    split_sources = {
        "shot_zone_basic": "shot_zone_basic",
        "shot_zone_area": "shot_zone_area",
        "shot_zone_range": "shot_zone_range",
        "distance_band": "distance_band",
    }

    frames: list[pd.DataFrame] = []

    for split_type, source_col in split_sources.items():
        if source_col not in raw_shots_df.columns:
            continue

        grouped = (
            raw_shots_df.groupby(base_cols + [source_col], dropna=False)
            .agg(
                attempts=("shot_attempted_flag", "sum"),
                makes=("shot_made_flag", "sum"),
            )
            .reset_index()
            .rename(columns={source_col: "split_value"})
        )

        grouped["split_type"] = split_type
        grouped["fg_pct"] = (
            grouped["makes"] / grouped["attempts"].where(grouped["attempts"] != 0, pd.NA) * 100
        ).fillna(0).round(1)

        frames.append(grouped[base_cols + ["split_type", "split_value", "attempts", "makes", "fg_pct"]])

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["season", "player_name", "split_type", "attempts"], ascending=[False, True, True, False]).reset_index(drop=True)
    return out


def aggregate_shot_profile_overall(raw_shots_df: pd.DataFrame) -> pd.DataFrame:
    if raw_shots_df.empty:
        return pd.DataFrame()

    df = raw_shots_df.copy()
    base_cols = ["player_id", "player_name", "team_abbreviation", "season"]

    summary = (
        df.groupby(base_cols)
        .agg(
            total_shot_attempts=("shot_attempted_flag", "sum"),
            total_shot_makes=("shot_made_flag", "sum"),
            avg_shot_distance=("shot_distance", "mean"),
        )
        .reset_index()
    )
    summary["avg_shot_distance"] = summary["avg_shot_distance"].fillna(0).round(1)

    zone_map = {
        "rim": df["shot_zone_basic"].eq("Restricted Area"),
        "midrange": df["shot_zone_basic"].isin(["Mid-Range", "In The Paint (Non-RA)"]),
        "corner3": df["shot_zone_basic"].isin(["Left Corner 3", "Right Corner 3"]),
        "above_break3": df["shot_zone_basic"].eq("Above the Break 3"),
    }

    for label, mask in zone_map.items():
        zone_df = df[mask].copy()

        grouped = (
            zone_df.groupby(base_cols)
            .agg(
                zone_attempts=("shot_attempted_flag", "sum"),
                zone_makes=("shot_made_flag", "sum"),
            )
            .reset_index()
        )

        grouped[f"{label}_attempts"] = grouped["zone_attempts"]
        grouped[f"{label}_fg_pct"] = (
            grouped["zone_makes"] / grouped["zone_attempts"].where(grouped["zone_attempts"] != 0, pd.NA) * 100
        ).fillna(0).round(1)

        grouped = grouped[base_cols + [f"{label}_attempts", f"{label}_fg_pct"]]
        summary = summary.merge(grouped, on=base_cols, how="left")

    summary = summary.fillna(0)

    for label in ["rim", "midrange", "corner3", "above_break3"]:
        attempts_col = f"{label}_attempts"
        rate_col = f"{label}_attempt_rate"
        summary[rate_col] = (
            summary[attempts_col] / summary["total_shot_attempts"].where(summary["total_shot_attempts"] != 0, pd.NA)
        ).fillna(0).round(4)

    summary = summary.sort_values(["season", "total_shot_attempts"], ascending=[False, False]).reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Semantic document builders
# ---------------------------------------------------------------------------


def _season_summary_doc(row: pd.Series) -> dict:
    player_name = row["player_name"]
    team = row["team_abbreviation"]
    season = row["season"]

    ppg = _safe_float(row.get("points_per_game", 0), 1)
    rpg = _safe_float(row.get("rebounds_per_game", 0), 1)
    apg = _safe_float(row.get("assists_per_game", 0), 1)
    gp = _safe_int(row.get("games_played", 0))
    fg_pct = _safe_float(row.get("fg_pct", 0), 1)
    fg3_pct = _safe_float(row.get("fg3_pct", 0), 1)
    ft_pct = _safe_float(row.get("ft_pct", 0), 1)

    text = (
        f"In the {season} season, {player_name} ({team}) averaged {ppg} points, {rpg} rebounds, "
        f"and {apg} assists per game over {gp} games. "
        f"He shot {fg_pct}% from the field, {fg3_pct}% from three, and {ft_pct}% from the free-throw line."
    )

    return _doc(text, player_name, team, season, "season_summary")


def _season_style_doc(row: pd.Series) -> dict:
    player_name = row["player_name"]
    team = row["team_abbreviation"]
    season = row["season"]

    ppg = _safe_float(row.get("points_per_game", 0), 1)
    rpg = _safe_float(row.get("rebounds_per_game", 0), 1)
    apg = _safe_float(row.get("assists_per_game", 0), 1)
    fg3a = _safe_float(row.get("fg3a_per_game", 0), 1)
    fg3pct = _safe_float(row.get("fg3_pct", 0), 1)

    scorer_label = (
        "high-volume scorer" if ppg >= 25
        else "secondary scorer" if ppg >= 18
        else "lower-volume scorer"
    )
    playmaker_label = (
        "strong playmaker" if apg >= 6
        else "secondary playmaker" if apg >= 4
        else "limited playmaker"
    )
    rebounding_label = (
        "strong rebounder" if rpg >= 8
        else "solid rebounder" if rpg >= 5
        else "lighter rebounder"
    )
    shooting_label = (
        "high-volume three-point shooter" if fg3a >= 7
        else "moderate three-point shooter" if fg3a >= 4
        else "lower-volume three-point shooter"
    )
    efficiency_label = (
        "efficient from three" if fg3pct >= 38
        else "solid from three" if fg3pct >= 35
        else "inconsistent from three"
    )

    text = (
        f"In the {season} season, {player_name} ({team}) profiled as a {scorer_label}, "
        f"{playmaker_label}, and {rebounding_label}. "
        f"He was a {shooting_label} and was {efficiency_label}. "
        f"His season averages were {ppg} points, {rpg} rebounds, and {apg} assists per game."
    )

    return _doc(text, player_name, team, season, "season_style")


def _recent_summary_doc(player_name: str, team: str, season: str, player_games: pd.DataFrame) -> dict:
    games_used = len(player_games)
    avg_points = player_games["points"].mean() if "points" in player_games.columns else 0
    avg_rebounds = player_games["rebounds"].mean() if "rebounds" in player_games.columns else 0
    avg_assists = player_games["assists"].mean() if "assists" in player_games.columns else 0
    avg_fg_pct = player_games["fg_pct"].mean() if "fg_pct" in player_games.columns else 0
    avg_fg3_pct = player_games["fg3_pct"].mean() if "fg3_pct" in player_games.columns else 0

    text = (
        f"Over his last {games_used} games of the {season} season, {player_name} ({team}) averaged "
        f"{_safe_float(avg_points, 1)} points, {_safe_float(avg_rebounds, 1)} rebounds, and "
        f"{_safe_float(avg_assists, 1)} assists per game. "
        f"Over that span he shot {_safe_float(avg_fg_pct, 1)}% from the field and "
        f"{_safe_float(avg_fg3_pct, 1)}% from three."
    )
    return _doc(text, player_name, team, season, "recent_summary")


def _recent_game_doc(row: pd.Series) -> dict:
    player_name = row["player_name"]
    team = row["team_abbreviation"]
    season = row["season"]

    text = (
        f"On {row.get('game_date', 'a recent date')} in the {season} season, {player_name} ({team}) in matchup "
        f"{row.get('matchup', 'N/A')} recorded {_safe_int(row.get('points', 0))} points, "
        f"{_safe_int(row.get('rebounds', 0))} rebounds, {_safe_int(row.get('assists', 0))} assists, "
        f"{_safe_int(row.get('steals', 0))} steals, and {_safe_int(row.get('blocks', 0))} blocks. "
        f"He shot {_safe_float(row.get('fg_pct', 0), 1)}% from the field and "
        f"{_safe_float(row.get('fg3_pct', 0), 1)}% from three."
    )
    return _doc(text, player_name, team, season, "recent_game")


def _shot_profile_doc(row: pd.Series) -> dict:
    player_name = row["player_name"]
    team = row["team_abbreviation"]
    season = row["season"]

    text = (
        f"In the {season} season, {player_name} ({team}) took {_safe_int(row.get('attempts', 0))} shots in "
        f"{row.get('split_value', 'unknown')} for split type {row.get('split_type', 'unknown')}, "
        f"making {_safe_int(row.get('makes', 0))} for {_safe_float(row.get('fg_pct', 0), 1)}% shooting."
    )
    return _doc(text, player_name, team, season, "shot_profile_split")


def _shot_profile_overall_doc(row: pd.Series) -> dict:
    player_name = row["player_name"]
    team = row["team_abbreviation"]
    season = row["season"]

    text = (
        f"In the {season} season, {player_name} ({team}) had a shot profile with "
        f"{_safe_int(row.get('total_shot_attempts', 0))} tracked attempts and an average shot distance of "
        f"{_safe_float(row.get('avg_shot_distance', 0), 1)} feet. "
        f"He took {round(_safe_float(row.get('rim_attempt_rate', 0), 4) * 100, 1)}% of his shots at the rim, "
        f"{round(_safe_float(row.get('midrange_attempt_rate', 0), 4) * 100, 1)}% from mid-range, "
        f"{round(_safe_float(row.get('corner3_attempt_rate', 0), 4) * 100, 1)}% from the corners, and "
        f"{round(_safe_float(row.get('above_break3_attempt_rate', 0), 4) * 100, 1)}% above the break. "
        f"He shot {_safe_float(row.get('rim_fg_pct', 0), 1)}% at the rim, "
        f"{_safe_float(row.get('midrange_fg_pct', 0), 1)}% from mid-range, "
        f"{_safe_float(row.get('corner3_fg_pct', 0), 1)}% on corner threes, and "
        f"{_safe_float(row.get('above_break3_fg_pct', 0), 1)}% on above-the-break threes."
    )
    return _doc(text, player_name, team, season, "shot_profile_overall")


def _shot_style_doc(row: pd.Series) -> dict:
    player_name = row["player_name"]
    team = row["team_abbreviation"]
    season = row["season"]

    rim_rate = _safe_float(row.get("rim_attempt_rate", 0), 4) * 100
    mid_rate = _safe_float(row.get("midrange_attempt_rate", 0), 4) * 100
    corner_rate = _safe_float(row.get("corner3_attempt_rate", 0), 4) * 100
    above_rate = _safe_float(row.get("above_break3_attempt_rate", 0), 4) * 100

    dominant_zone = max(
        [
            ("rim", rim_rate),
            ("mid-range", mid_rate),
            ("corners", corner_rate),
            ("above the break", above_rate),
        ],
        key=lambda x: x[1],
    )[0]

    text = (
        f"In the {season} season, {player_name} ({team}) was primarily a {dominant_zone}-oriented shooter. "
        f"His shot mix was {rim_rate:.1f}% at the rim, {mid_rate:.1f}% from mid-range, "
        f"{corner_rate:.1f}% from the corners, and {above_rate:.1f}% above the break."
    )
    return _doc(text, player_name, team, season, "shot_style")


def build_documents(
    season_stats_df: pd.DataFrame,
    recent_games_df: pd.DataFrame,
    shot_profile_df: pd.DataFrame,
    shot_profile_overall_df: pd.DataFrame,
) -> list[dict]:
    docs: list[dict] = []

    if not season_stats_df.empty:
        for _, row in season_stats_df.iterrows():
            docs.append(_season_summary_doc(row))
            docs.append(_season_style_doc(row))

    if not recent_games_df.empty:
        for _, row in recent_games_df.iterrows():
            docs.append(_recent_game_doc(row))

        grouped_recent = (
            recent_games_df.sort_values(["season", "player_name", "game_date"], ascending=[False, True, False])
            .groupby(["player_name", "team_abbreviation", "season"])
        )
        for (player_name, team, season), group in grouped_recent:
            docs.append(_recent_summary_doc(player_name, team, season, group.head(5).copy()))

    if not shot_profile_df.empty:
        shot_profile_filtered = shot_profile_df[shot_profile_df["attempts"] >= 15].copy()
        for _, row in shot_profile_filtered.iterrows():
            docs.append(_shot_profile_doc(row))

    if not shot_profile_overall_df.empty:
        for _, row in shot_profile_overall_df.iterrows():
            docs.append(_shot_profile_overall_doc(row))
            docs.append(_shot_style_doc(row))

    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    seasons = DEFAULT_SEASONS

    season_stats_df = fetch_all_season_player_stats(seasons=seasons)
    recent_games_df = fetch_all_recent_player_games(
        season_stats_df=season_stats_df,
        seasons=seasons,
        top_n_players=DEFAULT_TOP_N_RECENT,
        games_per_player=5,
    )
    raw_shots_df = fetch_all_raw_shots(
        season_stats_df=season_stats_df,
        seasons=seasons,
        top_n_players=DEFAULT_TOP_N_SHOT_PLAYERS,
    )
    shot_profile_df = aggregate_shot_profile(raw_shots_df)
    shot_profile_overall_df = aggregate_shot_profile_overall(raw_shots_df)

    _write_csv(season_stats_df, "season_player_stats.csv")
    _write_csv(recent_games_df, "recent_player_games.csv")
    _write_csv(raw_shots_df, "raw_shots.csv")
    _write_csv(shot_profile_df, "player_shot_profile.csv")
    _write_csv(shot_profile_overall_df, "player_shot_profile_overall.csv")

    docs = build_documents(
        season_stats_df=season_stats_df,
        recent_games_df=recent_games_df,
        shot_profile_df=shot_profile_df,
        shot_profile_overall_df=shot_profile_overall_df,
    )
    _write_json(docs, "nba_docs.json")

    print("[ingest] Done.")


if __name__ == "__main__":
    main()