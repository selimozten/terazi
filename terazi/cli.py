"""CLI entry point for terazi."""

from __future__ import annotations

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
def generate(category: str, num_examples: int, batch_size: int, output_dir: str, region: str) -> None:
    """Generate benchmark data using Claude via AWS Bedrock."""
    categories = ["core", "tool", "fin", "legal"] if category == "all" else [category]
    output_path = Path(output_dir)

    for cat in categories:
        module_path, class_name = GENERATORS[cat].rsplit(":", 1)
        import importlib
        module = importlib.import_module(module_path)
        generator_cls = getattr(module, class_name)
        generator = generator_cls(output_dir=output_path, region=region)
        generator.generate(num_examples=num_examples, batch_size=batch_size)


@cli.command("eval")
@click.option("--model", "-m", required=True, help="Model name (HF model ID or API model)")
@click.option("--categories", "-c", type=str, default="core,tool,fin,legal")
@click.option("--backend", type=click.Choice(["hf", "api"]), default="hf")
@click.option("--base-url", type=str, default=None, help="API base URL (for api backend)")
@click.option("--api-key", type=str, default=None, help="API key (for api backend)")
@click.option("--data-dir", type=click.Path(), default="data")
@click.option("--results-dir", type=click.Path(), default="results")
def evaluate(
    model: str,
    categories: str,
    backend: str,
    base_url: str | None,
    api_key: str | None,
    data_dir: str,
    results_dir: str,
) -> None:
    """Run a model against terazi benchmarks."""
    from terazi.eval.runner import APIBackend, EvalRunner, HFBackend, print_results

    cat_list = [c.strip() for c in categories.split(",")]

    if backend == "hf":
        model_backend = HFBackend(model)
    else:
        model_backend = APIBackend(model, base_url=base_url or "https://api.openai.com/v1", api_key=api_key)

    runner = EvalRunner(data_dir=Path(data_dir), results_dir=Path(results_dir))
    results = runner.run(model_backend, model, cat_list)
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
