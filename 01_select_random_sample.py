#!/usr/bin/env python3
"""Select a reproducible random sample of eligible pilot responses."""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path


DEFAULT_INPUT = Path("pilot1_calls.csv")
DEFAULT_OUTPUT = Path("selected_20.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample rows whose response_status and parse_status are both "
            "'ok'. The complete source rows are retained for annotation."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=20, help="Number of rows to select.")
    parser.add_argument(
        "--seed",
        type=int,
        default=20260819,
        help="Random seed (default: 20260819).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be positive")
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.output.exists() and not args.force:
        raise SystemExit(
            f"Output already exists: {args.output}. Use --force to replace it."
        )

    with args.input.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"response_id", "response_status", "parse_status", "reason"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise SystemExit(f"Input is missing required columns: {', '.join(missing)}")

        eligible = [
            row
            for row in reader
            if row["response_status"].strip().lower() == "ok"
            and row["parse_status"].strip().lower() == "ok"
        ]

    response_ids = [row["response_id"] for row in eligible]
    duplicate_ids = sorted(
        response_id
        for response_id in set(response_ids)
        if response_ids.count(response_id) > 1
    )
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise SystemExit(f"Eligible response_id values are not unique (e.g. {preview})")
    if len(eligible) < args.n:
        raise SystemExit(
            f"Requested {args.n} rows, but only {len(eligible)} rows are eligible."
        )

    rng = random.Random(args.seed)
    selected = rng.sample(eligible, args.n)
    output_fields = ["selection_order", *fieldnames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()
            for order, row in enumerate(selected, start=1):
                writer.writerow({"selection_order": order, **row})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        f"Selected {len(selected)} of {len(eligible)} eligible rows "
        f"(seed={args.seed}) -> {args.output}"
    )
    print("response_id values:", ", ".join(row["response_id"] for row in selected))


if __name__ == "__main__":
    main()
