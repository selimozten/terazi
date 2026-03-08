"""Aggregate eval results into a leaderboard data.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CATEGORIES = ["core", "tool", "fin", "legal"]


def parse_result_filename(filename: str) -> tuple[str, str, int] | None:
    """Parse '{model_safe}_{category}_{timestamp}.json' right-to-left."""
    stem = Path(filename).stem
    parts = stem.rsplit("_", 2)
    if len(parts) < 3:
        return None
    model_safe, category, ts_str = parts[0], parts[1], parts[2]
    if category not in CATEGORIES:
        # category might contain underscores — try again
        parts = stem.rsplit("_", 2)
        if len(parts) < 3:
            return None
        *model_parts, category, ts_str = parts
        model_safe = "_".join(model_parts)
    if category not in CATEGORIES:
        return None
    try:
        timestamp = int(ts_str)
    except ValueError:
        return None
    return model_safe, category, timestamp


def build_leaderboard(results_dir: Path) -> dict[str, Any]:
    """Scan results dir, deduplicate, strip examples, return leaderboard dict."""
    result_files = sorted(results_dir.glob("*.json"))

    # Collect all results keyed by (model, category), keep latest timestamp
    best: dict[tuple[str, str], tuple[int, dict]] = {}
    all_subcategories: dict[str, set[str]] = {c: set() for c in CATEGORIES}

    for f in result_files:
        parsed = parse_result_filename(f.name)
        if parsed is None:
            continue
        _, category, timestamp = parsed

        with open(f) as fh:
            data = json.load(fh)

        model_name = data.get("model", "unknown")
        key = (model_name, category)

        if key not in best or timestamp > best[key][0]:
            best[key] = (timestamp, data)

        # Collect subcategory names
        per_sub = data.get("per_subcategory", {})
        all_subcategories[category].update(per_sub.keys())

    # Group by model
    model_results: dict[str, dict[str, Any]] = {}
    for (model_name, category), (timestamp, data) in best.items():
        if model_name not in model_results:
            model_results[model_name] = {}

        per_sub = data.get("per_subcategory", {})
        per_subcategory = {
            sub: round(info["mean"], 4) if isinstance(info, dict) else round(info, 4)
            for sub, info in per_sub.items()
        }

        model_results[model_name][category] = {
            "overall": round(data["scores"]["overall"], 4),
            "total": data["total"],
            "timestamp": timestamp,
            "per_subcategory": per_subcategory,
        }

    # Build models list with averages
    models = []
    for name, results in model_results.items():
        scores = [results[c]["overall"] for c in CATEGORIES if c in results]
        average = round(sum(scores) / len(scores), 4) if scores else 0.0
        models.append({
            "name": name,
            "results": results,
            "average": average,
        })

    # Sort by average descending
    models.sort(key=lambda m: m["average"], reverse=True)

    subcategories = {
        c: sorted(subs) for c, subs in all_subcategories.items() if subs
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "categories": CATEGORIES,
        "subcategories": subcategories,
        "models": models,
    }


def write_leaderboard(results_dir: Path, output: Path) -> dict[str, Any]:
    """Build leaderboard and write to output path."""
    data = build_leaderboard(results_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
