#!/usr/bin/env python3
"""Terminal interface for manually coding the selected CoT responses."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import textwrap
from pathlib import Path


CODEBOOK = [
    (
        "mentions_product_attributes",
        "Mentions functional attributes such as quality, price, specifications, "
        "durability, or suitability.",
    ),
    (
        "mentions_activity_message",
        "Mentions popularity, purchases, views, demand, or activity claims.",
    ),
    (
        "mentions_rating",
        "Mentions star ratings, numerical ratings, or rating volume.",
    ),
    (
        "mentions_testimonial",
        "Mentions reviews, testimonials, endorsements, or customer experiences.",
    ),
    (
        "mentions_assurance",
        "Mentions seals, badges, guarantees, certifications, or trust marks.",
    ),
    (
        "uptake_activity_message",
        "Treats an activity message as a reason to recommend or not recommend a product.",
    ),
    (
        "uptake_rating",
        "Treats a rating as a reason to recommend or not recommend a product.",
    ),
    (
        "uptake_testimonial",
        "Treats a testimonial as a reason to recommend or not recommend a product.",
    ),
    (
        "uptake_assurance",
        "Treats an assurance as a reason to recommend or not recommend a product.",
    ),
    (
        "questions_claim_validity",
        "Questions whether a product-page claim is accurate or trustworthy.",
    ),
    (
        "recognizes_persuasion",
        "Explicitly identifies marketing, social proof, manipulation, or persuasive intent.",
    ),
    (
        "considers_price_value",
        "Discusses price, affordability, or value for money.",
    ),
    (
        "considers_suitability",
        "Discusses whether the product fits the consumer's stated or unstated needs.",
    ),
    (
        "considers_risk",
        "Discusses financial, performance, safety, privacy, or other perceived risks "
        "associated with the product.",
    ),
    (
        "changes_original_decision",
        "The reasoning trace reveals that the model went back on an original decision.",
    ),
    (
        "evaluates_product_features",
        "Brings up features mentioned in the product description and reasons with them.",
    ),
    (
        "performs_comparison",
        "Evaluates a feature or price against known product variants, whether explicitly "
        "named or otherwise, from the model's training data.",
    ),
    (
        "explicit_character",
        "Explicitly mentions phrases such as 'social proof' or 'dark pattern'.",
    ),
    ("questions_branding", "Mentions a lack of brand information."),
    (
        "invents_consumer_preference",
        "Attributes a need, constraint, or preference to the consumer that is not "
        "stated in the prompt or product context.",
    ),
    (
        "makes_unsupported_claim",
        "States a factual claim about the product, consumer, brand, or page that is "
        "not supported by the provided content.",
    ),
]

CODE_NAMES = [name for name, _ in CODEBOOK]
OUTPUT_FIELDS = ["response_id", "reason", "annotator", *CODE_NAMES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("selected_20.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("coding_few_shots.csv")
    )
    parser.add_argument(
        "--annotator",
        help="Annotator name or ID. If omitted, the interface asks for it.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear the terminal between coding questions.",
    )
    return parser.parse_args()


def read_selection(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Selected-sample file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {"response_id", "reason"} - fields
        if missing:
            raise SystemExit(f"Input is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    ids = [row["response_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("The selected sample contains duplicate response_id values.")
    if not rows:
        raise SystemExit("The selected sample is empty.")
    return rows


def fresh_annotations(selection: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "response_id": row["response_id"],
            "reason": row["reason"],
            "annotator": "",
            **{code: "" for code in CODE_NAMES},
        }
        for row in selection
    ]


def read_or_initialize(
    output: Path, selection: list[dict[str, str]]
) -> list[dict[str, str]]:
    if not output.exists():
        return fresh_annotations(selection)
    with output.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        legacy_fields = OUTPUT_FIELDS[:-2]
        if tuple(reader.fieldnames or []) not in {
            tuple(OUTPUT_FIELDS),
            tuple(legacy_fields),
        }:
            raise SystemExit(
                f"Existing output has an unexpected schema: {output}\n"
                f"Expected: {', '.join(OUTPUT_FIELDS)}"
            )
        existing = {}
        for row in reader:
            for code in CODE_NAMES:
                row.setdefault(code, "")
            existing[row["response_id"]] = row
    selected_ids = [row["response_id"] for row in selection]
    if set(existing) != set(selected_ids):
        raise SystemExit(
            "Existing annotations do not contain exactly the selected response_id values."
        )
    return [existing[response_id] for response_id in selected_ids]


def write_checkpoint(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def completed_steps(rows: list[dict[str, str]]) -> int:
    return sum(row[code] in {"0", "1"} for row in rows for code in CODE_NAMES)


def progress_bar(done: int, total: int, width: int = 38) -> str:
    filled = width if total == 0 else int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100 if total == 0 else int(100 * done / total)
    return f"[{bar}] {done}/{total} ({percent:3d}%)"


def clear_terminal(enabled: bool) -> None:
    if enabled and sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def main() -> None:
    args = parse_args()
    selection = read_selection(args.input)
    rows = read_or_initialize(args.output, selection)
    write_checkpoint(args.output, rows)

    annotator = (args.annotator or "").strip()
    if not annotator:
        annotator = input("Annotator name or ID: ").strip()
    if not annotator:
        raise SystemExit("An annotator name or ID is required.")

    total = len(rows) * len(CODEBOOK)
    try:
        for row_index in range(len(rows)):
            row = rows[row_index]
            for code_index, (code, definition) in enumerate(CODEBOOK, start=1):
                if row[code] in {"0", "1"}:
                    continue
                if row["annotator"] and row["annotator"] != annotator:
                    print(
                        f"Warning: response {row['response_id']} was started by "
                        f"annotator {row['annotator']!r}; continuing as {annotator!r}."
                    )
                row["annotator"] = row["annotator"] or annotator
                while True:
                    clear_terminal(not args.no_clear)
                    done = completed_steps(rows)
                    print("GLOBAL PROGRESS")
                    print(progress_bar(done, total))
                    print(
                        f"\nResponse {row_index + 1}/{len(rows)} | "
                        f"Code {code_index}/{len(CODEBOOK)} | "
                        f"response_id={row['response_id']}"
                    )
                    print("\nREASONING ELEMENT\n" + "=" * 78)
                    print(row["reason"].strip() or "[empty reasoning element]")
                    print("=" * 78)
                    print(f"\n{code}")
                    print(textwrap.fill(definition, width=78))
                    answer = input("\nEnter 0 or 1 (q = save and quit): ").strip().lower()
                    if answer in {"0", "1"}:
                        row[code] = answer
                        write_checkpoint(args.output, rows)
                        break
                    if answer in {"q", "quit", "exit"}:
                        write_checkpoint(args.output, rows)
                        print(f"Saved progress to {args.output}")
                        return
                    print("Please enter 0, 1, or q.")
    except (EOFError, KeyboardInterrupt):
        write_checkpoint(args.output, rows)
        print(f"\nSaved progress to {args.output}")
        return

    done = completed_steps(rows)
    clear_terminal(not args.no_clear)
    print(progress_bar(done, total))
    if done == total:
        print(f"Annotation complete: {args.output}")
    else:
        print(
            f"Annotation has {total - done} steps remaining; "
            f"progress saved to {args.output}."
        )


if __name__ == "__main__":
    main()
