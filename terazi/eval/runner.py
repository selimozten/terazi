"""Evaluation runner: load benchmark, run inference, score results."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from terazi.eval.formats import load_jsonl
from terazi.eval.metrics import get_metric_fn
from terazi.eval.prompts import get_system_prompt

console = Console()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CONCURRENCY = 20


class EvalResult(BaseModel):
    model: str
    category: str
    total: int
    scores: dict[str, float]
    per_subcategory: dict[str, dict[str, float]]
    examples: list[dict[str, Any]] = []


class ModelBackend:
    """Base class for model inference backends."""

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        raise NotImplementedError

    async def agenerate(self, prompt: str, system_prompt: str = "") -> str:
        return self.generate(prompt, system_prompt)


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

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class APIBackend(ModelBackend):
    """OpenAI-compatible API backend (works with OpenRouter, OpenAI, etc.)."""

    def __init__(
        self,
        model_name: str,
        base_url: str = OPENROUTER_BASE_URL,
        api_key: str | None = None,
        max_tokens: int = 2048,
        max_retries: int = 5,
    ) -> None:
        import openai

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.client = openai.OpenAI(base_url=base_url, api_key=resolved_key)
        self.async_client = openai.AsyncOpenAI(base_url=base_url, api_key=resolved_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def _build_messages(self, prompt: str, system_prompt: str) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = self._build_messages(prompt, system_prompt)
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt
                    time.sleep(delay)
                    continue
                raise
        return ""

    async def agenerate(self, prompt: str, system_prompt: str = "") -> str:
        messages = self._build_messages(prompt, system_prompt)
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt
                    await asyncio.sleep(delay)
                    continue
                raise
        return ""


class EvalRunner:
    """Run a model against terazi benchmarks and score results."""

    def __init__(
        self,
        data_dir: Path = Path("data"),
        results_dir: Path = Path("results"),
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.concurrency = concurrency

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

            result = asyncio.run(
                self._eval_category(backend, model_name, category, data_file, difficulty, sample)
            )
            results.append(result)
            self._save_result(result)

        return results

    async def _eval_category(
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

        console.print(f"\n[bold]Evaluating {model_name} on terazi-{category}[/bold]")
        console.print(f"  Examples: {len(examples)}, concurrency: {self.concurrency}")

        semaphore = asyncio.Semaphore(self.concurrency)
        completed = 0

        async def process_example(ex: dict[str, Any]) -> dict[str, Any]:
            nonlocal completed
            subcat = ex.get("subcategory", "unknown")
            metric_fn = metric_fns.get(subcat, metric_fns.get(list(metric_fns.keys())[0]))
            system_prompt = get_system_prompt(category, subcat)

            async with semaphore:
                try:
                    predicted = await backend.agenerate(ex["input"], system_prompt)
                except Exception as e:
                    console.print(f"[red]Error on {ex['id']}: {e}[/red]")
                    predicted = ""

            score = metric_fn(predicted, ex["expected_output"])
            completed += 1
            if completed % 50 == 0 or completed == len(examples):
                console.print(f"  [{completed}/{len(examples)}]")

            return {
                "id": ex["id"],
                "subcategory": subcat,
                "score": score,
                "predicted": predicted,
                "expected": ex["expected_output"],
            }

        tasks = [process_example(ex) for ex in examples]
        scored_examples = await asyncio.gather(*tasks)

        subcategory_scores: dict[str, list[float]] = {}
        for se in scored_examples:
            subcategory_scores.setdefault(se["subcategory"], []).append(se["score"])

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
