"""Scoring functions for terazi evaluation."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable


def exact_match(predicted: str, expected: str) -> float:
    """Return 1.0 if predicted matches expected after normalization, else 0.0."""
    return 1.0 if _normalize(predicted) == _normalize(expected) else 0.0


def choice_match(predicted: str, expected: str) -> float:
    """Match multiple-choice answers by extracting the answer letter/prefix."""
    pred_choice = _extract_choice(predicted)
    exp_choice = _extract_choice(expected)
    if pred_choice and exp_choice:
        return 1.0 if pred_choice == exp_choice else 0.0
    # Fallback to F1 if no choice letter found
    return f1_score(predicted, expected)


def sentiment_match(predicted: str, expected: str) -> float:
    """Match sentiment labels (pozitif/negatif/nötr) extracted from text."""
    pred_sent = _extract_sentiment(predicted)
    exp_sent = _extract_sentiment(expected)
    if pred_sent and exp_sent:
        return 1.0 if pred_sent == exp_sent else 0.0
    return 0.0


def f1_score(predicted: str, expected: str) -> float:
    """Token-level F1 score for extraction tasks."""
    pred_tokens = _normalize(predicted).split()
    exp_tokens = _normalize(expected).split()
    if not pred_tokens or not exp_tokens:
        return 1.0 if pred_tokens == exp_tokens else 0.0

    common = Counter(pred_tokens) & Counter(exp_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu(predicted: str, expected: str) -> float:
    """Sentence-level BLEU score using sacrebleu."""
    if not predicted.strip() or not expected.strip():
        return 1.0 if predicted.strip() == expected.strip() else 0.0
    import sacrebleu

    result = sacrebleu.sentence_bleu(predicted, [expected])
    return result.score / 100.0


def rouge_l(predicted: str, expected: str) -> float:
    """ROUGE-L F1 score."""
    if not predicted.strip() or not expected.strip():
        return 1.0 if predicted.strip() == expected.strip() else 0.0
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(expected, predicted)
    return scores["rougeL"].fmeasure


def tool_call_match(predicted: str, expected: str) -> float:
    """Match tool calls: function name and parameters with partial credit."""
    try:
        pred = _parse_tool_call(predicted)
        exp = _parse_tool_call(expected)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
        return 0.0

    if not isinstance(pred, dict) or not isinstance(exp, dict):
        return 0.0

    if pred.get("tool") != exp.get("tool"):
        return 0.0

    pred_params = pred.get("params", {})
    exp_params = exp.get("params", {})
    if pred_params == exp_params:
        return 1.0

    # Partial credit: fraction of correct parameters
    if not exp_params:
        return 1.0 if not pred_params else 0.5
    correct = sum(1 for k, v in exp_params.items() if pred_params.get(k) == v)
    return correct / len(exp_params)


MetricFn = Callable[[str, str], float]


def get_metric_fn(category: str) -> dict[str, MetricFn]:
    """Return the appropriate metric functions for a benchmark category."""
    metric_map: dict[str, dict[str, MetricFn]] = {
        "core": {
            "reading_comprehension": f1_score,
            "common_sense": choice_match,
            "grammar": f1_score,
            "translation": bleu,
            "summarization": rouge_l,
        },
        "tool": {
            "api_call": tool_call_match,
            "multi_step": tool_call_match,
            "parameter_extraction": tool_call_match,
            "error_recovery": f1_score,
        },
        "fin": {
            "document_comprehension": f1_score,
            "sentiment": sentiment_match,
            "numerical_reasoning": f1_score,
            "term_understanding": f1_score,
        },
        "legal": {
            "document_comprehension": f1_score,
            "case_reasoning": f1_score,
            "clause_extraction": f1_score,
            "regulatory_compliance": f1_score,
        },
    }
    return metric_map.get(category, {})


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _extract_choice(text: str) -> str | None:
    """Extract multiple-choice answer letter (A, B, C, D) from text."""
    text = text.strip()
    # Match patterns like "B)", "B.", "Cevap: B", "Doğru cevap B)"
    match = re.search(r"\b([A-D])\s*[\)\.:]", text)
    if match:
        return match.group(1).upper()
    # Try "Cevap: X" or "cevap X"
    match = re.search(r"(?:cevap|doğru|yanıt)[:\s]*([A-D])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Just a single letter at the start
    match = re.match(r"^([A-D])\b", text)
    if match:
        return match.group(1).upper()
    return None


def _extract_sentiment(text: str) -> str | None:
    """Extract sentiment label from text."""
    text = text.strip().lower()
    for label in ["pozitif", "negatif", "nötr", "notr"]:
        if label in text:
            return "nötr" if label in ("nötr", "notr") else label
    return None


def _parse_tool_call(text: str) -> dict[str, Any]:
    text = text.strip()
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{") and not text.startswith("["):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group()
    parsed = json.loads(text)
    # Handle list of tool calls (multi_step) — score first call
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    return parsed
