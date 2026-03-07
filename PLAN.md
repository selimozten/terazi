# terazi

Comprehensive benchmark suite for Turkish language models.

"Terazi" means "scales" in Turkish — the instrument used for weighing and measuring.

## Why This Exists

There is no standardized, comprehensive benchmark for Turkish LLMs. Anyone training a Turkish model today has no reliable way to measure if it's actually good. terazi fixes that.

## Benchmark Categories

### terazi-core
General Turkish language understanding and generation.
- Reading comprehension
- Common sense reasoning
- Grammar and linguistics
- Translation quality (TR<->EN)
- Summarization

### terazi-tool
Tool use and function calling in Turkish.
- API call generation from Turkish instructions
- Multi-step tool chains
- Parameter extraction from natural language
- Error recovery and clarification

### terazi-fin
Financial Turkish language tasks.
- Financial document comprehension (BIST filings, KAP disclosures)
- Sentiment analysis on Turkish financial news
- Numerical reasoning from Turkish financial reports
- Financial term understanding

### terazi-legal
Legal Turkish language tasks.
- Turkish legal document comprehension
- Case law reasoning
- Contract clause extraction
- Regulatory compliance questions

## Data Generation Pipeline

1. Use Claude Opus 4.6 via AWS Bedrock to generate high-quality evaluation data
2. $9,000 AWS credits available for generation
3. Pipeline:
   - Define task templates and schemas for each category
   - Generate candidate questions + reference answers via Opus
   - Human review pass for quality (spot-check + fix)
   - Format as standardized eval harness compatible dataset
4. Target: 500-1000 examples per category, 2000-4000 total

## Eval Harness

- Compatible with EleutherAI lm-evaluation-harness
- Also provide standalone eval script (no heavy dependencies)
- Metrics: accuracy, F1, BLEU/ROUGE where applicable, tool-call exact match
- Leaderboard: static site or HuggingFace Space showing model rankings

## Output Artifacts

- HuggingFace dataset: `selimozten/terazi`
- Eval harness config files
- Leaderboard (GitHub Pages or HF Space)
- Paper/blog post documenting methodology

## Tech Stack

- Python for eval harness and data generation scripts
- AWS Bedrock SDK (boto3) for Opus API calls
- HuggingFace datasets library for publishing
- GitHub Actions for CI on eval harness

## Project Structure

```
terazi/
  README.md
  setup.py / pyproject.toml
  terazi/
    __init__.py
    generate/          # Data generation scripts (Bedrock/Opus)
      core.py
      tool.py
      fin.py
      legal.py
    eval/              # Evaluation harness
      runner.py
      metrics.py
      formats.py
    data/              # Generated benchmark data (or HF download)
      core/
      tool/
      fin/
      legal/
  configs/             # lm-evaluation-harness task configs
  leaderboard/         # Static site for results
  scripts/
    generate_all.sh
    run_eval.sh
```

## Milestones

1. Repo setup, project structure, generation pipeline scaffolding
2. terazi-core: generate + validate core Turkish benchmark
3. terazi-tool: generate tool-use evaluation set
4. terazi-fin: generate financial Turkish benchmark
5. terazi-legal: generate legal Turkish benchmark
6. Eval harness: standalone runner + lm-eval-harness integration
7. Run baseline evals (GPT-4, Claude, Gemini, open Turkish models)
8. Publish dataset on HuggingFace + leaderboard
9. Write methodology blog post / paper

## Success Criteria

- Cited/used by at least one other Turkish LLM project
- Baseline results for 5+ models on the leaderboard
- Clean, reproducible eval pipeline anyone can run
