"""
ingest.py
---------
Pulls raw NBA data via nba_api and converts player / team stats into
semantic narrative documents ready for embedding.

Each document is a self-contained English sentence or paragraph that
describes one stat line, so the retriever can surface relevant context
for natural-language questions like "Who dominates rebounds this week?".

Usage
-----
    python -m rag.ingest          # writes docs to data/processed_docs/
"""

import os
import json
import time
import pandas as pd

from nba_api.stats.endpoints import (
    LeagueLeaders,
    PlayerGameLog,
    LeagueDashPlayerStats,
)
from nba_api.stats.static import players

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val, decimals: int = 1) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return 0.0


def _player_season_narrative(row: pd.Series) -> str:
    """
    Convert a single season-average row into a natural-language sentence.

    Example output:
        "LeBron James averages 25.3 points, 7.8 rebounds, and 8.1 assists
         per game this season while shooting 54.2% from the field."
    """
    name = row.get("PLAYER_NAME", "Unknown")
    pts = _safe_float(row.get("PTS", 0))
    reb = _safe_float(row.get("REB", 0))
    ast = _safe_float(row.get("AST", 0))
    stl = _safe_float(row.get("STL", 0))
    blk = _safe_float(row.get("BLK", 0))
    fg_pct = _safe_float(row.get("FG_PCT", 0) * 100)
    fg3_pct = _safe_float(row.get("FG3_PCT", 0) * 100)
    gp = int(row.get("GP", 0))
    team = row.get("TEAM_ABBREVIATION", "N/A")

    return (
        f"{name} ({team}) averages {pts} points, {reb} rebounds, and {ast} assists "
        f"per game this season over {gp} games, while also contributing {stl} steals "
        f"and {blk} blocks. He shoots {fg_pct}% from the field and {fg3_pct}% "
        f"from three-point range."
    )


def _recent_game_narrative(player_name: str, game_row: pd.Series, game_num: int) -> str:
    """
    Produce a narrative for a single recent game log entry.

    Example output:
        "In his most recent game (game 1 of last 5), Nikola Jokic scored
         32 points, grabbed 13 rebounds, and dished 10 assists."
    """
    pts = int(_safe_float(game_row.get("PTS", 0)))
    reb = int(_safe_float(game_row.get("REB", 0)))
    ast = int(_safe_float(game_row.get("AST", 0)))
    stl = int(_safe_float(game_row.get("STL", 0)))
    blk = int(_safe_float(game_row.get("BLK", 0)))
    matchup = game_row.get("MATCHUP", "vs. opponent")
    date = game_row.get("GAME_DATE", "recently")

    ordinal = {1: "most recent", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(
        game_num, f"game {game_num}"
    )

    return (
        f"In his {ordinal} game of his last 5 ({date}, {matchup}), "
        f"{player_name} scored {pts} points, grabbed {reb} rebounds, and dished "
        f"{ast} assists with {stl} steals and {blk} blocks."
    )


# ---------------------------------------------------------------------------
# Main ingest functions
# ---------------------------------------------------------------------------

def ingest_season_averages(season: str = "2024-25") -> list[str]:
    """
    Pull season-average leaders and convert to narrative documents.

    Returns a list of document strings.
    """
    print(f"[ingest] Fetching season averages for {season}...")
    leaders = LeagueLeaders(season=season, stat_category_abbreviation="PTS")
    df = leaders.get_data_frames()[0]

    # Also pull full stat dashboard for richer context
    time.sleep(1)
    dash = LeagueDashPlayerStats(season=season)
    df_full = dash.get_data_frames()[0]

    docs = []
    for _, row in df_full.iterrows():
        docs.append(_player_season_narrative(row))

    print(f"[ingest] Generated {len(docs)} season-average documents.")
    return docs


def ingest_recent_games(top_n_players: int = 50, season: str = "2024-25") -> list[str]:
    """
    For the top-N players by scoring, pull their last 5 game logs
    and convert each game into a narrative.

    Returns a list of document strings.
    """
    print(f"[ingest] Fetching recent game logs for top {top_n_players} players...")

    leaders = LeagueLeaders(season=season, stat_category_abbreviation="PTS")
    df = leaders.get_data_frames()[0].head(top_n_players)

    all_docs: list[str] = []

    for _, player_row in df.iterrows():
        player_id = player_row["PLAYER_ID"]
        player_name = player_row["PLAYER_NAME"]

        try:
            time.sleep(0.6)  # respect rate limits
            log = PlayerGameLog(player_id=player_id, season=season)
            games_df = log.get_data_frames()[0].head(5)

            for i, (_, game) in enumerate(games_df.iterrows(), start=1):
                all_docs.append(_recent_game_narrative(player_name, game, i))

        except Exception as exc:
            print(f"  [warn] Could not fetch games for {player_name}: {exc}")

    print(f"[ingest] Generated {len(all_docs)} recent-game documents.")
    return all_docs


def save_documents(docs: list[str], filename: str = "nba_docs.json") -> str:
    """Persist document list as JSON and return the file path."""
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w") as f:
        json.dump(docs, f, indent=2)
    print(f"[ingest] Saved {len(docs)} documents → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    season_docs = ingest_season_averages()
    game_docs = ingest_recent_games(top_n_players=50)
    all_docs = season_docs + game_docs
    save_documents(all_docs)
    print("[ingest] Done.")
