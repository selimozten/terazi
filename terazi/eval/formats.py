"""Data format utilities for loading and converting benchmark data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(
    path: Path,
    difficulty: str | None = None,
    subcategory: str | None = None,
) -> list[dict[str, Any]]:
    """Load examples from a JSONL file, optionally filtering by difficulty or subcategory."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if difficulty and ex.get("difficulty") != difficulty:
                continue
            if subcategory and ex.get("subcategory") != subcategory:
                continue
            examples.append(ex)
    return examples


def save_jsonl(examples: list[dict[str, Any]], path: Path) -> None:
    """Save examples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def to_lm_eval_format(examples: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    """Convert terazi examples to lm-evaluation-harness format."""
    converted = []
    for ex in examples:
        converted.append({
            "doc_id": ex["id"],
            "query": ex["input"],
            "choices": [],
            "gold": ex["expected_output"],
            "task": f"terazi_{category}",
        })
    return converted


def to_hf_dataset(examples: list[dict[str, Any]]) -> dict[str, list]:
    """Convert terazi examples to HuggingFace datasets column format."""
    columns: dict[str, list] = {
        "id": [],
        "category": [],
        "subcategory": [],
        "difficulty": [],
        "input": [],
        "expected_output": [],
    }
    for ex in examples:
        for key in columns:
            columns[key].append(ex.get(key, ""))
    return columns
