"""
query_router.py
---------------
Lightweight query classifier for deciding whether a question should be answered
from structured stats or from semantic retrieval (FAISS).

Adds season extraction so structured answers can hard-filter by season.
If no season is mentioned, downstream code should assume the most recent
completed season (2024-25).
"""

from __future__ import annotations

import re

LATEST_COMPLETED_SEASON = "2024-25"


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(p in text for p in phrases)


def extract_season_reference(query: str) -> str | None:
    q = _normalize(query)

    # explicit season like 2023-24
    match = re.search(r"\b(20\d{2}-\d{2})\b", q)
    if match:
        return match.group(1)

    if "this season" in q or "current season" in q:
        return LATEST_COMPLETED_SEASON

    if "last season" in q:
        return "2023-24"

    return None


def classify_query(query: str) -> dict:
    q = _normalize(query)

    asks_top = _has_any(q, [
        "best", "top", "highest", "most", "leader", "leaders"
    ])

    asks_compare = (
        _has_any(q, [
            "compare", "vs", "versus", "better than",
            "different from", "difference between", "differ"
        ])
        or (
            _has_any(q, ["how do", "how does", "what is the difference between", "what's the difference between"])
            and _has_any(q, [" and ", " differ", " different"])
        )
    )

    mentions_recent = _has_any(q, [
        "last 5", "last five", "recent", "recently", "lately"
    ])

    mentions_shooting_zone = _has_any(q, [
        "corner", "corners", "corner 3", "corner three",
        "above the break", "restricted area",
        "mid-range", "midrange",
        "paint", "rim",
        "shot profile", "shot zone",
        "from three", "from 3", "3-point", "three-point"
    ])

    asks_shooter = _has_any(q, [
        "best shooter", "top shooter", "better shooter", "shoots best"
    ])

    asks_recent_form = _has_any(q, [
        "how has", "how is", "how was", "how's", "hows"
    ]) and _has_any(q, [
        "been playing", "performed", "playing lately", "playing recently", "played lately", "played recently"
    ])

    asks_player_summary = _has_any(q, [
        "how good is", "tell me about", "what kind of player is",
        "what kind of player", "summarize", "is "
    ])

    stat = None

    if _has_any(q, ["3 point", "three point", "3pt", "3-point", "from three", "from 3"]):
        stat = "fg3_pct"
    elif _has_any(q, ["free throw", "ft%"]):
        stat = "ft_pct"
    elif _has_any(q, ["field goal percentage", "fg%", "shooting percentage", "best shooter", "top shooter", "shoots best"]):
        stat = "fg_pct"
    elif _has_any(q, ["points", "score", "scoring"]):
        stat = "points"
    elif _has_any(q, ["rebounds", "rebounding"]):
        stat = "rebounds"
    elif _has_any(q, ["assists"]):
        stat = "assists"
    elif _has_any(q, ["steals"]):
        stat = "steals"
    elif _has_any(q, ["blocks"]):
        stat = "blocks"

    zone = None
    if _has_any(q, ["left corner 3", "left corner three"]):
        zone = "Left Corner 3"
    elif _has_any(q, ["right corner 3", "right corner three"]):
        zone = "Right Corner 3"
    elif _has_any(q, ["corner 3", "corner three", "corners", "from the corners", "from corners", "corner shots", "corner shooting"]):
        zone = "corner3_combined"
    elif _has_any(q, ["above the break"]):
        zone = "Above the Break 3"
    elif _has_any(q, ["restricted area", "rim"]):
        zone = "Restricted Area"
    elif _has_any(q, ["mid-range", "midrange"]):
        zone = "Mid-Range"
    elif _has_any(q, ["paint", "in the paint"]):
        zone = "In The Paint (Non-RA)"

    timeframe = None
    if _has_any(q, ["last 5", "last five"]):
        timeframe = "last_5"
    elif _has_any(q, ["season", "this season", "last season"]):
        timeframe = "season"
    elif mentions_recent:
        timeframe = "recent"

    season = extract_season_reference(query)

    if asks_compare:
        return {
            "route": "structured",
            "intent": "player_comparison",
            "stat": stat,
            "timeframe": timeframe,
            "season": season,
            "zone": zone,
            "comparison": True,
        }

    if asks_recent_form:
        return {
            "route": "structured",
            "intent": "player_recent_summary",
            "stat": stat,
            "timeframe": timeframe or "last_5",
            "season": season,
            "zone": None,
            "comparison": False,
        }

    if asks_player_summary and not asks_top and not asks_compare and not mentions_shooting_zone:
        return {
            "route": "structured",
            "intent": "player_summary",
            "stat": stat,
            "timeframe": timeframe or "season",
            "season": season,
            "zone": None,
            "comparison": False,
        }

    if (asks_top or asks_shooter) and mentions_shooting_zone:
        return {
            "route": "structured",
            "intent": "shot_zone_leaderboard",
            "stat": stat or "fg_pct",
            "timeframe": timeframe or "season",
            "season": season,
            "zone": zone,
            "comparison": False,
        }

    if asks_top and timeframe == "last_5":
        return {
            "route": "structured",
            "intent": "recent_leaderboard",
            "stat": stat or "points",
            "timeframe": "last_5",
            "season": season,
            "zone": None,
            "comparison": False,
        }

    if asks_top and stat in {"points", "rebounds", "assists", "steals", "blocks", "fg3_pct", "fg_pct", "ft_pct"}:
        return {
            "route": "structured",
            "intent": "season_leaderboard",
            "stat": stat,
            "timeframe": timeframe or "season",
            "season": season,
            "zone": None,
            "comparison": False,
        }

    if mentions_shooting_zone:
        return {
            "route": "structured",
            "intent": "shot_zone_lookup",
            "stat": stat or "fg_pct",
            "timeframe": timeframe or "season",
            "season": season,
            "zone": zone,
            "comparison": False,
        }

    return {
        "route": "semantic",
        "intent": "semantic_general",
        "stat": stat,
        "timeframe": timeframe,
        "season": season,
        "zone": zone,
        "comparison": False,
    }