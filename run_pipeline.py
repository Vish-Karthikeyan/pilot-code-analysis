#!/usr/bin/env python3
"""Interactive, resumable runner for the complete DP CoT analysis pipeline."""

from __future__ import annotations

import argparse
import csv
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PILOT_FILE = ROOT / "pilot1_calls.csv"
SAMPLE_FILE = ROOT / "selected_20.csv"
ANNOTATION_FILE = ROOT / "coding_few_shots.csv"
CODEBOOK_FILE = ROOT / "pilot_codebook.csv"
ENV_FILE = ROOT / ".env"

CODE_NAMES = [
    "mentions_product_attributes",
    "mentions_activity_message",
    "mentions_rating",
    "mentions_testimonial",
    "mentions_assurance",
    "uptake_activity_message",
    "uptake_rating",
    "uptake_testimonial",
    "uptake_assurance",
    "questions_claim_validity",
    "recognizes_persuasion",
    "considers_price_value",
    "considers_suitability",
    "considers_risk",
    "changes_original_decision",
    "evaluates_product_features",
    "performs_comparison",
    "explicit_character",
    "questions_branding",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run selection, human annotation, optional OpenRouter coding, and "
            "optional Stata regressions. Existing outputs are resumed."
        )
    )
    parser.add_argument(
        "--seed", type=int, default=20260819, help="Random-sample seed."
    )
    parser.add_argument(
        "--annotator", help="Annotator name/ID; prompted for when omitted."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status and exit without changing anything.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def eligible_pilot_ids() -> set[str]:
    fields, rows = read_rows(PILOT_FILE)
    required = {"response_id", "response_status", "parse_status"}
    if not required.issubset(fields):
        return set()
    return {
        row["response_id"]
        for row in rows
        if row["response_status"].strip().lower() == "ok"
        and row["parse_status"].strip().lower() == "ok"
    }


def sample_count() -> int:
    fields, rows = read_rows(SAMPLE_FILE)
    return len(rows) if {"response_id", "reason"}.issubset(fields) else 0


def annotation_progress() -> tuple[int, int, int]:
    fields, rows = read_rows(ANNOTATION_FILE)
    if not {"response_id", "reason", *CODE_NAMES}.issubset(fields):
        return 0, 0, 20 * len(CODE_NAMES)
    completed = sum(
        row.get(code) in {"0", "1"} for row in rows for code in CODE_NAMES
    )
    total = len(rows) * len(CODE_NAMES)
    completed_rows = sum(
        all(row.get(code) in {"0", "1"} for code in CODE_NAMES) for row in rows
    )
    return completed_rows, completed, total


def coded_ids() -> set[str]:
    fields, rows = read_rows(CODEBOOK_FILE)
    if not {"response_id", "reason", *CODE_NAMES}.issubset(fields):
        return set()
    valid: set[str] = set()
    for row in rows:
        if row["response_id"] and all(row.get(code) in {"0", "1"} for code in CODE_NAMES):
            valid.add(row["response_id"])
    return valid


def dotenv_value(name: str) -> str | None:
    if not ENV_FILE.is_file():
        return None
    for raw_line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value or None
    return None


def find_stata() -> str | None:
    configured = os.environ.get("STATA_BIN")
    if configured and Path(configured).expanduser().is_file():
        return str(Path(configured).expanduser())

    for command in ("stata-mp", "stata-se", "stata", "StataMP", "StataSE", "Stata"):
        found = shutil.which(command)
        if found:
            return found

    application_candidates = [
        "/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp",
        "/Applications/Stata/StataSE.app/Contents/MacOS/stata-se",
        "/Applications/Stata/StataBE.app/Contents/MacOS/stata",
        "/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp",
        "/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se",
    ]
    for candidate in application_candidates:
        if Path(candidate).is_file():
            return candidate
    return None


def ask_yes_no(prompt: str, *, default: bool) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        answer = input(prompt + suffix).strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}\n", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def show_status() -> None:
    eligible = eligible_pilot_ids()
    annotated_rows, annotated_steps, annotation_total = annotation_progress()
    coded = coded_ids()
    configured_key = bool(
        os.environ.get("OPENROUTER_API_KEY") or dotenv_value("OPENROUTER_API_KEY")
    )
    stata = find_stata()

    print("DP CoT pipeline status")
    print(f"  Eligible pilot rows:       {len(eligible)}")
    print(f"  Selected sample rows:      {sample_count()}/20")
    print(
        f"  Human annotation:         {annotated_rows}/20 rows "
        f"({annotated_steps}/{annotation_total} code decisions)"
    )
    print(f"  LLM-coded eligible rows:   {len(coded)}/{len(eligible)}")
    print(f"  OpenRouter key configured: {'yes' if configured_key else 'no'}")
    print(f"  Stata detected:            {stata or 'no'}")


def main() -> int:
    args = parse_args()
    if args.status:
        show_status()
        return 0
    if not sys.stdin.isatty():
        print("This runner is interactive and must be launched in a terminal.", file=sys.stderr)
        return 2
    if not PILOT_FILE.is_file():
        print(f"Missing required input: {PILOT_FILE}", file=sys.stderr)
        return 2

    print("DP Chain-of-Thought Analysis Pipeline")
    print("=====================================")
    print("Existing outputs are detected and resumed automatically.\n")

    if sample_count() == 0:
        run(
            [
                sys.executable,
                str(ROOT / "01_select_random_sample.py"),
                "--seed",
                str(args.seed),
            ]
        )
    elif sample_count() != 20:
        print(
            f"{SAMPLE_FILE.name} exists but does not contain 20 valid rows. "
            "Move or repair it before rerunning.",
            file=sys.stderr,
        )
        return 2
    else:
        print(f"Selection already complete: {SAMPLE_FILE.name}")

    annotated_rows, annotated_steps, annotation_total = annotation_progress()
    if annotated_steps < annotation_total or annotated_rows < 20:
        annotation_command = [
            sys.executable,
            str(ROOT / "02_annotate_few_shots.py"),
        ]
        if args.annotator:
            annotation_command.extend(["--annotator", args.annotator])
        run(annotation_command)
        annotated_rows, annotated_steps, annotation_total = annotation_progress()

    if annotated_rows < 20 or annotated_steps < annotation_total:
        print(
            f"\nAnnotation is saved at {annotated_steps}/{annotation_total} decisions."
        )
        print("Rerun this command to continue. Later stages require all 20 rows.")
        return 0
    print("Human annotation complete: 20/20 rows.")

    eligible = eligible_pilot_ids()
    coded = coded_ids()
    if coded != eligible:
        print(
            "\nOptional LLM coding can label all eligible pilot rows through OpenRouter."
        )
        print(
            "This makes paid API calls using openai/gpt-5.6-sol. The process is "
            "resumable, but review OpenRouter pricing before continuing."
        )
        if not ask_yes_no("Run the LLM coding step now?", default=False):
            print("Stopping after human annotation. No API key is required for this path.")
            return 0

        api_key = os.environ.get("OPENROUTER_API_KEY") or dotenv_value(
            "OPENROUTER_API_KEY"
        )
        if not api_key:
            api_key = getpass.getpass(
                "Enter your OpenRouter API key (input is hidden; blank cancels): "
            ).strip()
        if not api_key:
            print("No API key supplied. Stopping after human annotation.")
            return 0
        if not ask_yes_no(
            f"Confirm paid coding of up to {len(eligible) - len(coded)} remaining rows?",
            default=False,
        ):
            print("LLM coding cancelled.")
            return 0

        child_env = os.environ.copy()
        child_env["OPENROUTER_API_KEY"] = api_key
        run([sys.executable, str(ROOT / "03_llm_code_pilot.py")], env=child_env)
        coded = coded_ids()

    if coded != eligible:
        print(
            f"LLM coding is incomplete ({len(coded)}/{len(eligible)}). "
            "Rerun this command to resume."
        )
        return 1
    print(f"LLM coding complete: {len(coded)}/{len(eligible)} eligible rows.")

    stata = find_stata()
    if not stata:
        print("\nStata was not detected. The pipeline is complete through LLM coding.")
        print(
            "To run regressions later, install Stata or set STATA_BIN to its "
            "executable and rerun this command."
        )
        return 0

    print(f"\nOptional Stata installation detected: {stata}")
    if not ask_yes_no("Run the Section 1 regressions now?", default=True):
        print("Stopping before Stata. Rerun this command whenever you are ready.")
        return 0
    run([stata, "-b", "do", str(ROOT / "04_run_regressions.do")])
    print("\nPipeline complete. Regression outputs are in stata_results/.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(
            f"\nA pipeline step exited with status {exc.returncode}. "
            "Its completed output is preserved; rerun to resume.",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed output has been preserved.", file=sys.stderr)
        raise SystemExit(130)
