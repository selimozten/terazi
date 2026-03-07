"""Evaluation runner: load benchmark, run inference, score results."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from terazi.eval.formats import load_jsonl
from terazi.eval.metrics import get_metric_fn

console = Console()


class EvalResult(BaseModel):
    model: str
    category: str
    total: int
    scores: dict[str, float]
    per_subcategory: dict[str, dict[str, float]]
    examples: list[dict[str, Any]] = []


class ModelBackend:
    """Base class for model inference backends."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class HFBackend(ModelBackend):
    """HuggingFace transformers backend."""

    def __init__(self, model_name: str, max_new_tokens: int = 512) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class APIBackend(ModelBackend):
    """OpenAI-compatible API backend."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        import openai

        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content or ""


class EvalRunner:
    """Run a model against terazi benchmarks and score results."""

    def __init__(
        self,
        data_dir: Path = Path("data"),
        results_dir: Path = Path("results"),
    ) -> None:
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        backend: ModelBackend,
        model_name: str,
        categories: list[str] | None = None,
        difficulty: str | None = None,
        sample: int | None = None,
    ) -> list[EvalResult]:
        all_categories = categories or ["core", "tool", "fin", "legal"]
        results = []

        for category in all_categories:
            data_file = self.data_dir / category / f"{category}.jsonl"
            if not data_file.exists():
                console.print(f"[yellow]No data for {category}, skipping[/yellow]")
                continue

            result = self._eval_category(backend, model_name, category, data_file, difficulty, sample)
            results.append(result)
            self._save_result(result)

        return results

    def _eval_category(
        self,
        backend: ModelBackend,
        model_name: str,
        category: str,
        data_file: Path,
        difficulty: str | None = None,
        sample: int | None = None,
    ) -> EvalResult:
        import random

        examples = load_jsonl(data_file, difficulty=difficulty)
        if sample and sample < len(examples):
            examples = random.sample(examples, sample)
        metric_fns = get_metric_fn(category)

        scored_examples = []
        subcategory_scores: dict[str, list[float]] = {}

        console.print(f"\n[bold]Evaluating {model_name} on terazi-{category}[/bold]")
        console.print(f"  Examples: {len(examples)}")

        with Progress(console=console) as progress:
            task = progress.add_task(f"terazi-{category}", total=len(examples))

            for ex in examples:
                subcat = ex.get("subcategory", "unknown")
                metric_fn = metric_fns.get(subcat, metric_fns.get(list(metric_fns.keys())[0]))

                predicted = backend.generate(ex["input"])
                score = metric_fn(predicted, ex["expected_output"])

                subcategory_scores.setdefault(subcat, []).append(score)
                scored_examples.append({
                    "id": ex["id"],
                    "subcategory": subcat,
                    "score": score,
                    "predicted": predicted,
                    "expected": ex["expected_output"],
                })
                progress.advance(task)

        per_sub = {
            sub: {"mean": sum(s) / len(s), "count": len(s)}
            for sub, s in subcategory_scores.items()
        }
        all_scores = [s for scores in subcategory_scores.values() for s in scores]
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return EvalResult(
            model=model_name,
            category=category,
            total=len(examples),
            scores={"overall": overall},
            per_subcategory=per_sub,
            examples=scored_examples,
        )

    def _save_result(self, result: EvalResult) -> None:
        ts = int(time.time())
        safe_name = result.model.replace("/", "_")
        path = self.results_dir / f"{safe_name}_{result.category}_{ts}.json"
        with open(path, "w") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
        console.print(f"  Results saved to {path}")


def print_results(results: list[EvalResult]) -> None:
    """Print evaluation results as a table."""
    table = Table(title="terazi evaluation results")
    table.add_column("Category", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Examples", justify="right")

    for r in results:
        table.add_row(
            f"terazi-{r.category}",
            f"{r.scores['overall']:.3f}",
            str(r.total),
        )

    console.print(table)

    for r in results:
        sub_table = Table(title=f"terazi-{r.category} breakdown")
        sub_table.add_column("Subcategory")
        sub_table.add_column("Score", justify="right")
        sub_table.add_column("Count", justify="right")
        for sub, data in r.per_subcategory.items():
            sub_table.add_row(sub, f"{data['mean']:.3f}", str(int(data["count"])))
        console.print(sub_table)
