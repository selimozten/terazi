"""Base generator class for calling Claude Opus via AWS Bedrock."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

DEFAULT_MODEL_ID = "us.anthropic.claude-opus-4-6-20250515-v1:0"
DEFAULT_REGION = "us-east-1"
MAX_RETRIES = 5
BASE_DELAY = 2.0


class Example(BaseModel):
    id: str
    category: str
    subcategory: str
    difficulty: str
    input: str
    expected_output: str
    metadata: dict[str, Any] = {}


class BaseGenerator(ABC):
    """Base class for all benchmark data generators."""

    category: str = ""

    def __init__(
        self,
        output_dir: Path = Path("data"),
        model_id: str = DEFAULT_MODEL_ID,
        region: str = DEFAULT_REGION,
        max_tokens: int = 4096,
    ) -> None:
        self.output_dir = output_dir / self.category
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self._count = self._load_existing_count()

    def _load_existing_count(self) -> int:
        output_file = self.output_dir / f"{self.category}.jsonl"
        if not output_file.exists():
            return 0
        count = 0
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _call_bedrock(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude via Bedrock with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": self.max_tokens,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}],
                    }),
                )
                result = json.loads(response["body"].read())
                return result["content"][0]["text"]
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code in ("ThrottlingException", "TooManyRequestsException"):
                    delay = BASE_DELAY * (2**attempt)
                    console.print(f"[yellow]Rate limited, retrying in {delay:.0f}s...[/yellow]")
                    time.sleep(delay)
                    continue
                raise
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * (2**attempt)
                    console.print(f"[yellow]Error: {e}. Retrying in {delay:.0f}s...[/yellow]")
                    time.sleep(delay)
                    continue
                raise
        msg = f"Failed after {MAX_RETRIES} retries"
        raise RuntimeError(msg)

    def _save_example(self, example: Example) -> None:
        output_file = self.output_dir / f"{self.category}.jsonl"
        with open(output_file, "a") as f:
            f.write(example.model_dump_json() + "\n")

    def _make_id(self, index: int) -> str:
        return f"{self.category}-{index:04d}"

    def _parse_json_response(self, text: str) -> list[dict[str, Any]]:
        """Extract JSON array from model response, handling markdown fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)

    @abstractmethod
    def _get_system_prompt(self, subcategory: str) -> str:
        """Return the system prompt for generating examples of this subcategory."""
        ...

    @abstractmethod
    def _get_user_prompt(self, subcategory: str, batch_size: int) -> str:
        """Return the user prompt requesting a batch of examples."""
        ...

    @abstractmethod
    def _get_subcategories(self) -> list[str]:
        """Return the list of subcategories for this benchmark."""
        ...

    def generate(self, num_examples: int = 500, batch_size: int = 5) -> None:
        """Generate benchmark examples across all subcategories."""
        subcategories = self._get_subcategories()
        per_subcategory = num_examples // len(subcategories)
        remainder = num_examples % len(subcategories)

        console.print(f"[bold]Generating {num_examples} examples for terazi-{self.category}[/bold]")
        console.print(f"  Subcategories: {', '.join(subcategories)}")
        console.print(f"  ~{per_subcategory} examples each, batch size {batch_size}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for i, subcat in enumerate(subcategories):
                target = per_subcategory + (1 if i < remainder else 0)
                generated = 0
                task = progress.add_task(f"{subcat}: 0/{target}", total=target)

                system_prompt = self._get_system_prompt(subcat)

                while generated < target:
                    current_batch = min(batch_size, target - generated)
                    user_prompt = self._get_user_prompt(subcat, current_batch)

                    try:
                        response = self._call_bedrock(system_prompt, user_prompt)
                        examples_data = self._parse_json_response(response)
                    except (json.JSONDecodeError, RuntimeError) as e:
                        console.print(f"[red]Failed to parse response for {subcat}: {e}[/red]")
                        continue

                    for ex_data in examples_data:
                        if generated >= target:
                            break
                        self._count += 1
                        example = Example(
                            id=self._make_id(self._count),
                            category=self.category,
                            subcategory=subcat,
                            difficulty=ex_data.get("difficulty", "medium"),
                            input=ex_data["input"],
                            expected_output=ex_data["expected_output"],
                            metadata=ex_data.get("metadata", {}),
                        )
                        self._save_example(example)
                        generated += 1

                    progress.update(task, completed=generated, description=f"{subcat}: {generated}/{target}")

        console.print(f"[green]Done. Total examples: {self._count}[/green]")
        console.print(f"  Output: {self.output_dir / f'{self.category}.jsonl'}")
