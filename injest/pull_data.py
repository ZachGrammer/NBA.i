import json
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import leaguedashplayerstats

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "processed_data.json"

def pull_basic_player_stats(season: str = "2024-25") -> list[dict]:
    response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season"
    )

    df = response.get_data_frames()[0]

    keep_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "AGE",
        "GP",
        "W",
        "L",
        "MIN",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "TOV",
        "STL",
        "BLK",
        "BLKA",
        "PF",
        "PFD",
        "PTS",
        "PLUS_MINUS",
    ]

    available_cols = [col for col in keep_cols if col in df.columns]
    df = df[available_cols].copy()

    records = []
    for row in df.to_dict(orient="records"):
        record = {
            "season": season,
            "player_id": row.get("PLAYER_ID"),
            "player_name": row.get("PLAYER_NAME"),
            "team_id": row.get("TEAM_ID"),
            "team_abbreviation": row.get("TEAM_ABBREVIATION"),
            "basic_stats": {
                "age": row.get("AGE"),
                "games_played": row.get("GP"),
                "wins": row.get("W"),
                "losses": row.get("L"),
                "minutes_per_game": row.get("MIN"),
                "field_goals_made": row.get("FGM"),
                "field_goals_attempted": row.get("FGA"),
                "field_goal_pct": row.get("FG_PCT"),
                "three_pointers_made": row.get("FG3M"),
                "three_pointers_attempted": row.get("FG3A"),
                "three_point_pct": row.get("FG3_PCT"),
                "free_throws_made": row.get("FTM"),
                "free_throws_attempted": row.get("FTA"),
                "free_throw_pct": row.get("FT_PCT"),
                "offensive_rebounds": row.get("OREB"),
                "defensive_rebounds": row.get("DREB"),
                "rebounds": row.get("REB"),
                "assists": row.get("AST"),
                "turnovers": row.get("TOV"),
                "steals": row.get("STL"),
                "blocks": row.get("BLK"),
                "personal_fouls": row.get("PF"),
                "points": row.get("PTS"),
                "plus_minus": row.get("PLUS_MINUS"),
            },
        }
        records.append(record)

    return records


def save_json(data: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    data = pull_basic_player_stats()
    save_json(data, OUTPUT_PATH)
    print(f"Saved {len(data)} player records to {OUTPUT_PATH}")