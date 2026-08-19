# DP Chain-of-Thought Analysis Pipeline

This repository implements Section 1 of **CoT Analysis for DP Project**: reproducible sampling, terminal-based human annotation, optional LLM-assisted coding, and optional Stata regressions.

The pipeline is resumable. You can stop after annotation, after LLM coding, or after the complete Stata analysis depending on the tools and credentials available to you.

## Quick start

Requirements:

- Python 3.10 or newer. The Python workflow uses only the standard library.
- An [OpenRouter API key](https://openrouter.ai/settings/keys) for the optional LLM step.
- Stata 17 or newer for the optional regression step.

Clone the repository, enter its directory, and run:

```bash
python3 run_pipeline.py
```

The runner detects completed work and resumes automatically. It supports three paths:

1. **No API key and no Stata:** randomly select 20 rows and complete the human annotation interface.
2. **OpenRouter key but no Stata:** complete annotation and LLM-assisted coding of every eligible pilot row.
3. **OpenRouter key and Stata:** run the entire pipeline, including regressions.

When the LLM step is selected and no key is configured, the runner asks for the API key using hidden terminal input. The entered key is passed only to the child process and is not written to disk. You can alternatively create a local `.env` file:

```text
OPENROUTER_API_KEY=your-key-here
```

The `.env` file is ignored by Git. Never commit an API key.

The requested model is `openai/gpt-5.6-sol`. Coding all eligible rows makes paid API calls, so check [OpenRouter pricing](https://openrouter.ai/openai/gpt-5.6-sol/) before confirming that step.

## Pipeline stages

### 1. Random sample

`01_select_random_sample.py` selects 20 globally unique `response_id` values from `pilot1_calls.csv` where both `response_status` and `parse_status` equal `ok`. The default seed is `20260819`.

Output: `selected_20.csv`

### 2. Human annotation

`02_annotate_few_shots.py` displays each reasoning element while requesting a binary value for every one of the 19 codebook variables. It checkpoints after every decision and displays global progress over all 380 decisions.

Output: `coding_few_shots.csv`

One human annotator codes all 20 selected reasoning elements. The runner asks for the annotator's name or ID once and then proceeds through every remaining code decision:

```bash
python3 run_pipeline.py --annotator your_name
```

### 3. LLM-assisted coding (optional)

`03_llm_code_pilot.py` compiles the 20 completed human annotations as few-shot examples and codes all eligible pilot rows through OpenRouter. It requires strict JSON-schema output for the 19 binary variables, retries transient failures, checkpoints every successful row, and resumes without recoding completed IDs.

Outputs:

- `pilot_codebook.csv`
- `pilot_codebook_errors.jsonl` if any rows fail

### 4. Stata regressions (optional)

`04_run_regressions.do` merges `pilot1_calls.csv` and `pilot_codebook.csv` by `response_id`, then runs:

- One crossed-random-effects logistic regression for each binary code.
- The full mixed-effects recommendation-score model.
- The preliminary 19-code OLS model with robust standard errors.
- A 10-fold cross-validated elastic-net model.

The wrapper searches for Stata on `PATH` and in standard macOS application locations. For a custom installation, set `STATA_BIN`:

```bash
export STATA_BIN=/path/to/stata-mp
python3 run_pipeline.py
```

Output: `stata_results/`

## Check progress

This command performs no writes and makes no API calls:

```bash
python3 run_pipeline.py --status
```

## Run stages manually

```bash
python3 01_select_random_sample.py
python3 02_annotate_few_shots.py --annotator your_name
python3 03_llm_code_pilot.py
stata -b do 04_run_regressions.do
```

Use `--help` with any Python script for additional options. Generated data, Stata results, local environments, and secrets are excluded by `.gitignore`.

## Repository contents

- `CoT analysis for DP (1).pdf` - source specification.
- `pilot1_calls.csv` - pilot input data.
- `run_pipeline.py` - interactive end-to-end runner.
- `01_select_random_sample.py` - eligible-row sampling.
- `02_annotate_few_shots.py` - human coding interface.
- `03_llm_code_pilot.py` - OpenRouter coding protocol.
- `04_run_regressions.do` - Stata analysis.
