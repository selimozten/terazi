"""CLI entry point for terazi."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import click
from rich.console import Console

console = Console()

GENERATORS = {
    "core": "terazi.generate.core:CoreGenerator",
    "tool": "terazi.generate.tool:ToolGenerator",
    "fin": "terazi.generate.fin:FinGenerator",
    "legal": "terazi.generate.legal:LegalGenerator",
}


@click.group()
@click.version_option(package_name="terazi")
def cli() -> None:
    """terazi -- comprehensive benchmark suite for Turkish language models."""


@cli.command()
@click.option("--category", "-c", type=click.Choice(["core", "tool", "fin", "legal", "all"]), default="all")
@click.option("--num-examples", "-n", type=int, default=500)
@click.option("--batch-size", "-b", type=int, default=5)
@click.option("--output-dir", "-o", type=click.Path(), default="data")
@click.option("--region", type=str, default="us-east-1")
@click.option("--model-id", type=str, default=None, help="Override Bedrock model ID")
@click.option("--api-key", type=str, default=None, help="Bedrock API key (or set AWS_BEDROCK_API_KEY)")
def generate(
    category: str,
    num_examples: int,
    batch_size: int,
    output_dir: str,
    region: str,
    model_id: str | None,
    api_key: str | None,
) -> None:
    """Generate benchmark data using Claude via AWS Bedrock."""
    categories = ["core", "tool", "fin", "legal"] if category == "all" else [category]
    output_path = Path(output_dir)

    kwargs: dict = {"output_dir": output_path, "region": region}
    if model_id:
        kwargs["model_id"] = model_id
    if api_key:
        kwargs["api_key"] = api_key

    for cat in categories:
        module_path, class_name = GENERATORS[cat].rsplit(":", 1)
        module = importlib.import_module(module_path)
        generator_cls = getattr(module, class_name)
        generator = generator_cls(**kwargs)
        generator.generate(num_examples=num_examples, batch_size=batch_size)


@cli.command("eval")
@click.option("--model", "-m", required=True, help="Model name (e.g. google/gemini-2.5-flash)")
@click.option("--categories", "-c", type=str, default="core,tool,fin,legal")
@click.option("--backend", type=click.Choice(["hf", "api"]), default="api")
@click.option("--base-url", type=str, default=None, help="API base URL (default: OpenRouter)")
@click.option("--api-key", type=str, default=None, help="API key (or set OPENROUTER_API_KEY)")
@click.option("--data-dir", type=click.Path(), default="data")
@click.option("--results-dir", type=click.Path(), default="results")
@click.option("--max-tokens", type=int, default=2048, help="Max tokens for model response")
@click.option("--difficulty", type=click.Choice(["easy", "medium", "hard"]), default=None, help="Filter by difficulty")
@click.option("--sample", type=int, default=None, help="Run on a random subset of N examples")
@click.option("--concurrency", type=int, default=20, help="Number of concurrent API requests")
def evaluate(
    model: str,
    categories: str,
    backend: str,
    base_url: str | None,
    api_key: str | None,
    data_dir: str,
    results_dir: str,
    max_tokens: int,
    difficulty: str | None,
    sample: int | None,
    concurrency: int,
) -> None:
    """Run a model against terazi benchmarks."""
    from terazi.eval.runner import APIBackend, EvalRunner, HFBackend, print_results

    cat_list = [c.strip() for c in categories.split(",")]

    if backend == "hf":
        model_backend = HFBackend(model)
    else:
        kwargs: dict = {"model_name": model, "max_tokens": max_tokens}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        model_backend = APIBackend(**kwargs)

    runner = EvalRunner(data_dir=Path(data_dir), results_dir=Path(results_dir), concurrency=concurrency)
    results = runner.run(model_backend, model, cat_list, difficulty=difficulty, sample=sample)
    print_results(results)


@cli.command()
@click.option("--results-dir", type=click.Path(), default="results")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def results(results_dir: str, fmt: str) -> None:
    """Show evaluation results."""
    from rich.table import Table

    results_path = Path(results_dir)
    if not results_path.exists():
        console.print("[red]No results directory found.[/red]")
        return

    result_files = sorted(results_path.glob("*.json"))
    if not result_files:
        console.print("[yellow]No result files found.[/yellow]")
        return

    all_results = []
    for f in result_files:
        with open(f) as fh:
            all_results.append(json.load(fh))

    if fmt == "json":
        console.print_json(json.dumps(all_results, ensure_ascii=False, indent=2))
        return

    table = Table(title="terazi evaluation results")
    table.add_column("Model", style="bold")
    table.add_column("Category")
    table.add_column("Score", justify="right")
    table.add_column("Examples", justify="right")

    for r in all_results:
        table.add_row(
            r["model"],
            f"terazi-{r['category']}",
            f"{r['scores']['overall']:.3f}",
            str(r["total"]),
        )

    console.print(table)


@cli.command()
@click.option("--data-dir", "-d", type=click.Path(), default="data")
@click.option("--category", "-c", type=click.Choice(["core", "tool", "fin", "legal", "all"]), default="all")
def validate(data_dir: str, category: str) -> None:
    """Validate generated benchmark data for completeness and formatting."""
    from rich.table import Table

    from terazi.eval.formats import load_jsonl

    data_path = Path(data_dir)
    categories = ["core", "tool", "fin", "legal"] if category == "all" else [category]
    required_fields = {"id", "category", "subcategory", "difficulty", "input", "expected_output"}

    table = Table(title="data validation")
    table.add_column("Category")
    table.add_column("File")
    table.add_column("Examples", justify="right")
    table.add_column("Issues", justify="right")

    total_issues = 0
    for cat in categories:
        data_file = data_path / cat / f"{cat}.jsonl"
        if not data_file.exists():
            table.add_row(cat, str(data_file), "0", "[red]file missing[/red]")
            total_issues += 1
            continue

        examples = load_jsonl(data_file)
        issues = 0
        for i, ex in enumerate(examples):
            missing = required_fields - set(ex.keys())
            if missing:
                console.print(f"  [red]{cat} example {i}: missing fields {missing}[/red]")
                issues += 1
            if not ex.get("input", "").strip():
                console.print(f"  [red]{cat} example {i}: empty input[/red]")
                issues += 1
            if not ex.get("expected_output", "").strip():
                console.print(f"  [red]{cat} example {i}: empty expected_output[/red]")
                issues += 1

        status = f"[green]{issues}[/green]" if issues == 0 else f"[red]{issues}[/red]"
        table.add_row(cat, str(data_file), str(len(examples)), status)
        total_issues += issues

    console.print(table)
    if total_issues == 0:
        console.print("[green]All data valid.[/green]")
    else:
        console.print(f"[red]{total_issues} issue(s) found.[/red]")


@cli.command()
@click.option("--data-dir", "-d", type=click.Path(), default="data")
def stats(data_dir: str) -> None:
    """Show statistics about generated benchmark data."""
    from collections import Counter

    from rich.table import Table

    from terazi.eval.formats import load_jsonl

    data_path = Path(data_dir)
    grand_total = 0

    for cat in ["core", "tool", "fin", "legal"]:
        data_file = data_path / cat / f"{cat}.jsonl"
        if not data_file.exists():
            console.print(f"[dim]terazi-{cat}: no data[/dim]")
            continue

        examples = load_jsonl(data_file)
        grand_total += len(examples)

        subcat_counts = Counter(ex.get("subcategory", "unknown") for ex in examples)
        diff_counts = Counter(ex.get("difficulty", "unknown") for ex in examples)

        table = Table(title=f"terazi-{cat} ({len(examples)} examples)")
        table.add_column("Subcategory")
        table.add_column("Count", justify="right")
        for sub, count in sorted(subcat_counts.items()):
            table.add_row(sub, str(count))
        console.print(table)

        diff_parts = [f"{d}: {c}" for d, c in sorted(diff_counts.items())]
        console.print(f"  Difficulty: {', '.join(diff_parts)}\n")

    console.print(f"[bold]Total: {grand_total} examples[/bold]")


@cli.command()
@click.option("--data-dir", "-d", type=click.Path(), default="data")
@click.option("--category", "-c", type=click.Choice(["core", "tool", "fin", "legal"]), required=True)
@click.option("--format", "fmt", type=click.Choice(["lm-eval", "hf"]), required=True)
@click.option("--output", "-o", type=click.Path(), required=True)
def convert(data_dir: str, category: str, fmt: str, output: str) -> None:
    """Convert benchmark data to other formats."""
    from terazi.eval.formats import load_jsonl, save_jsonl, to_hf_dataset, to_lm_eval_format

    data_file = Path(data_dir) / category / f"{category}.jsonl"
    if not data_file.exists():
        console.print(f"[red]No data file: {data_file}[/red]")
        return

    examples = load_jsonl(data_file)
    output_path = Path(output)

    if fmt == "lm-eval":
        converted = to_lm_eval_format(examples, category)
        save_jsonl(converted, output_path)
    elif fmt == "hf":
        columns = to_hf_dataset(examples)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(columns, f, ensure_ascii=False, indent=2)

    console.print(f"[green]Converted {len(examples)} examples to {fmt} format: {output_path}[/green]")


@cli.command()
@click.option("--results-dir", type=click.Path(), default="results")
@click.option("--output", "-o", type=click.Path(), default="docs/data.json")
def leaderboard(results_dir: str, output: str) -> None:
    """Generate leaderboard data.json for the GitHub Pages site."""
    from rich.table import Table

    from terazi.leaderboard import write_leaderboard

    results_path = Path(results_dir)
    if not results_path.exists():
        console.print("[red]No results directory found.[/red]")
        return

    output_path = Path(output)
    data = write_leaderboard(results_path, output_path)

    table = Table(title="terazi leaderboard")
    table.add_column("#", justify="right")
    table.add_column("Model", style="bold")
    table.add_column("Average", justify="right")
    for cat in data["categories"]:
        table.add_column(cat.capitalize(), justify="right")

    for i, model in enumerate(data["models"], 1):
        row = [str(i), model["name"], f"{model['average']:.4f}"]
        for cat in data["categories"]:
            if cat in model["results"]:
                row.append(f"{model['results'][cat]['overall']:.4f}")
            else:
                row.append("--")
        table.add_row(*row)

    console.print(table)
    console.print(f"[green]Wrote {output_path} ({len(data['models'])} models)[/green]")
