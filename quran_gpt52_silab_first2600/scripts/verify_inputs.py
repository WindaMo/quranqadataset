#!/usr/bin/env python3
"""Verify the two prepared GPT-5.2 input files before API evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "first_2600"
EXPECTED = 2600

FILES = [
    DATA_DIR / "canonical_gold_qas_gpt52_SPTP_first2600.json",
    DATA_DIR / "canonical_gold_qas_gpt52_MPTP_first2600.json",
]


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Bukan JSON list: {path}")

    return data


failed = False

for path in FILES:
    data = load_json_list(path)

    keys = [
        (int(item["surah"]), int(item["ayat"]))
        for item in data
    ]

    missing_qa = sum(
        1
        for item in data
        if not isinstance(item.get("question"), str)
        or not item["question"].strip()
        or not isinstance(item.get("answer"), str)
        or not item["answer"].strip()
    )

    duplicate_count = len(keys) - len(set(keys))

    print("=" * 72)
    print(path.name)
    print(f"Jumlah data       : {len(data):,}")
    print(f"Jumlah ayat unik  : {len(set(keys)):,}")
    print(f"Duplikasi         : {duplicate_count:,}")
    print(f"Question/answer kosong: {missing_qa:,}")

    if (
        len(data) != EXPECTED
        or len(set(keys)) != EXPECTED
        or duplicate_count != 0
        or missing_qa != 0
    ):
        failed = True

if failed:
    raise SystemExit(
        "VERIFIKASI GAGAL. Jangan jalankan evaluasi SI-LAB."
    )

print("\nVERIFIKASI BERHASIL: SPTP dan MPTP masing-masing siap 2.600 data.")
