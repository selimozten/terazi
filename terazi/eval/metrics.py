"""Scoring functions for terazi evaluation."""

from __future__ import annotations

import json
import re
from collections import Counter


def exact_match(predicted: str, expected: str) -> float:
    """Return 1.0 if predicted matches expected after normalization, else 0.0."""
    return 1.0 if _normalize(predicted) == _normalize(expected) else 0.0


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
    import sacrebleu

    result = sacrebleu.sentence_bleu(predicted, [expected])
    return result.score / 100.0


def rouge_l(predicted: str, expected: str) -> float:
    """ROUGE-L F1 score."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(expected, predicted)
    return scores["rougeL"].fmeasure


def tool_call_match(predicted: str, expected: str) -> float:
    """Exact match on tool call: function name and parameters."""
    try:
        pred = _parse_tool_call(predicted)
        exp = _parse_tool_call(expected)
    except (json.JSONDecodeError, KeyError, TypeError):
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


def get_metric_fn(category: str):
    """Return the appropriate metric function for a benchmark category."""
    metric_map = {
        "core": {
            "reading_comprehension": f1_score,
            "common_sense": exact_match,
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
            "sentiment": exact_match,
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


def _parse_tool_call(text: str) -> dict:
    text = text.strip()
    if not text.startswith("{"):
        # Try to find JSON in text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group()
    return json.loads(text)
