# terazi

Comprehensive benchmark suite for Turkish language models.

## Why

There is no standardized, comprehensive benchmark for Turkish LLMs. terazi fixes that. The name means "scales" in Turkish -- the instrument used for weighing and measuring.

## Benchmark Categories

| Category | Description | Tasks |
|---|---|---|
| **terazi-core** | General Turkish language understanding | Reading comprehension, common sense, grammar, translation, summarization |
| **terazi-tool** | Tool use and function calling | API calls, multi-step chains, parameter extraction, error recovery |
| **terazi-fin** | Financial Turkish | Document comprehension, sentiment, numerical reasoning, terminology |
| **terazi-legal** | Legal Turkish | Document comprehension, case reasoning, clause extraction, regulatory compliance |

Target: 500-1000 examples per category, 2000-4000 total.

## Quick Start

### Install

```bash
pip install -e .
```

### Generate Benchmark Data

Requires AWS credentials configured for Bedrock access (Claude Opus).

```bash
# Generate all categories (500 examples each)
terazi generate --category all --num-examples 500

# Generate a specific category
terazi generate --category core --num-examples 100
```

### Run Evaluation

```bash
# Evaluate a HuggingFace model
terazi eval --model meta-llama/Llama-3.1-8B-Instruct --categories core,tool

# Evaluate an API model
terazi eval --model gpt-4 --backend api --base-url https://api.openai.com/v1

# View results
terazi results --format table
```

## Project Structure

```
terazi/
  terazi/
    generate/     Data generation pipeline (Bedrock/Opus)
    eval/         Evaluation harness (runner, metrics, formats)
  configs/        lm-evaluation-harness task configs
  scripts/        Shell scripts for generation and eval
  data/           Generated benchmark data (gitignored)
  results/        Evaluation results (gitignored)
```

## Adding Your Model

1. Generate benchmark data (or download from HuggingFace: `selimozten/terazi`)
2. Run the eval harness against your model
3. Submit results via PR to be added to the leaderboard

## License

MIT
