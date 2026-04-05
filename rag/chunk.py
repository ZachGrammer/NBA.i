import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed_data.json"
OUTPUT_PATH = BASE_DIR / "data" / "chunks.json"


def load_processed_data(input_path: Path) -> list[dict]:
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_player_chunk(record: dict) -> dict:
    basic_stats = record.get("basic_stats", {})

    season = record.get("season", "Unknown season")
    player_id = record.get("player_id")
    player_name = record.get("player_name", "Unknown player")
    team_id = record.get("team_id")
    team_abbreviation = record.get("team_abbreviation", "Unknown team")

    text = (
        f"{player_name} played for {team_abbreviation} in the {season} regular season. "
        f"He averaged {basic_stats.get('points', 'N/A')} points, "
        f"{basic_stats.get('rebounds', 'N/A')} rebounds, and "
        f"{basic_stats.get('assists', 'N/A')} assists per game. "
        f"He played {basic_stats.get('games_played', 'N/A')} games and averaged "
        f"{basic_stats.get('minutes_per_game', 'N/A')} minutes per game. "
        f"He shot {basic_stats.get('field_goal_pct', 'N/A')} from the field, "
        f"{basic_stats.get('three_point_pct', 'N/A')} from three, and "
        f"{basic_stats.get('free_throw_pct', 'N/A')} from the free throw line. "
        f"He averaged {basic_stats.get('steals', 'N/A')} steals, "
        f"{basic_stats.get('blocks', 'N/A')} blocks, and "
        f"{basic_stats.get('turnovers', 'N/A')} turnovers per game. "
        f"His plus minus was {basic_stats.get('plus_minus', 'N/A')}."
    )

    chunk_id = f"{player_id}_{season.replace('-', '_')}"

    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "player_id": player_id,
            "player_name": player_name,
            "team_id": team_id,
            "team_abbreviation": team_abbreviation,
            "season": season,
        },
    }


def chunk_player_data(records: list[dict]) -> list[dict]:
    chunks = []
    for record in records:
        chunks.append(build_player_chunk(record))
    return chunks


def save_chunks(chunks: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    records = load_processed_data(INPUT_PATH)
    chunks = chunk_player_data(records)
    save_chunks(chunks, OUTPUT_PATH)
    print(f"Saved {len(chunks)} chunks to {OUTPUT_PATH}")