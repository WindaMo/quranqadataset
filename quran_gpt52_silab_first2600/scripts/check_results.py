#!/usr/bin/env python3
"""Check completeness of GPT-5.2 SI-LAB SPTP and MPTP results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = 2600
VALID_VERDICTS = {"SUPPORTED", "NOT_SUPPORTED"}

CONFIGS = [
    {
        "name": "SPTP",
        "log": (
            ROOT
            / "outputs"
            / "sptp"
            / "validation_log_3models_gpt52_SPTP_first2600_silab.jsonl"
        ),
        "summary": (
            ROOT
            / "outputs"
            / "sptp"
            / "validation_summary_gpt52_SPTP_first2600_silab.json"
        ),
    },
    {
        "name": "MPTP",
        "log": (
            ROOT
            / "outputs"
            / "mptp"
            / "validation_log_3models_gpt52_MPTP_first2600_silab.jsonl"
        ),
        "summary": (
            ROOT
            / "outputs"
            / "mptp"
            / "validation_summary_gpt52_MPTP_first2600_silab.json"
        ),
    },
]


def inspect_log(path: Path) -> dict[str, Any]:
    valid_records: dict[tuple[int, int], dict[str, Any]] = {}
    invalid_records: list[dict[str, Any]] = []
    corrupted_lines = 0

    if not path.exists():
        return {
            "exists": False,
            "valid_unique": 0,
            "invalid_records": [],
            "corrupted_lines": 0,
        }

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                corrupted_lines += 1
                continue

            if record.get("event") != "validated":
                continue

            key = (
                record.get("surah"),
                record.get("ayat"),
            )
            verdicts = [
                record.get("A"),
                record.get("B"),
                record.get("C"),
            ]

            if all(v in VALID_VERDICTS for v in verdicts):
                valid_records[key] = record
            else:
                invalid_records.append(
                    {
                        "line": line_number,
                        "surah": record.get("surah"),
                        "ayat": record.get("ayat"),
                        "A": record.get("A"),
                        "B": record.get("B"),
                        "C": record.get("C"),
                    }
                )

    return {
        "exists": True,
        "valid_unique": len(valid_records),
        "invalid_records": invalid_records,
        "corrupted_lines": corrupted_lines,
    }


failed = False

for config in CONFIGS:
    result = inspect_log(config["log"])

    print("=" * 72)
    print(config["name"])
    print(f"Log ada              : {result['exists']}")
    print(f"Verdict valid unik   : {result['valid_unique']:,}")
    print(f"Event ERROR/EMPTY    : {len(result['invalid_records']):,}")
    print(f"Baris JSON rusak     : {result['corrupted_lines']:,}")
    print(f"Summary ada          : {config['summary'].exists()}")

    if (
        not result["exists"]
        or result["valid_unique"] != EXPECTED
        or result["corrupted_lines"] != 0
    ):
        failed = True

    if result["invalid_records"]:
        print("Contoh event tidak valid terakhir:")
        for record in result["invalid_records"][-5:]:
            print(record)

if failed:
    raise SystemExit(
        "\nBELUM SELESAI. Jalankan ulang run_sptp.ps1 atau run_mptp.ps1 "
        "sampai masing-masing memiliki 2.600 verdict valid."
    )

print(
    "\nSELESAI: SPTP dan MPTP masing-masing memiliki "
    "2.600 verdict SI-LAB yang valid."
)
