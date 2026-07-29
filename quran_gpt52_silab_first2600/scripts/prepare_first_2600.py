#!/usr/bin/env python3
"""
Take the first 2,600 evaluable GPT-5.2 QA records from each source file.

Important:
- SPTP and MPTP are selected independently.
- "Evaluable" means the record has a non-empty question and answer.
- This does NOT guarantee that the SPTP and MPTP verse sets are identical.
- Source files are never modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TARGET_SIZE = 2600
ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "source" / "gpt5"
OUTPUT_DIR = ROOT / "data" / "first_2600"


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Isi file harus berupa JSON list: {path}")

    return data


def is_evaluable_qa(item: dict[str, Any]) -> bool:
    question = item.get("question")
    answer = item.get("answer")

    return (
        isinstance(question, str)
        and bool(question.strip())
        and isinstance(answer, str)
        and bool(answer.strip())
        and item.get("surah") is not None
        and item.get("ayat") is not None
    )


def verse_key(item: dict[str, Any]) -> tuple[int, int]:
    try:
        return int(item["surah"]), int(item["ayat"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Nilai surah/ayat tidak valid pada item: {item}"
        ) from exc


def select_first_evaluable(
    data: list[dict[str, Any]],
    target_size: int,
    dataset_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    skipped_non_qa = 0
    skipped_duplicate = 0

    for item in data:
        if not is_evaluable_qa(item):
            skipped_non_qa += 1
            continue

        key = verse_key(item)

        if key in seen:
            skipped_duplicate += 1
            continue

        selected.append(item)
        seen.add(key)

        if len(selected) == target_size:
            break

    if len(selected) < target_size:
        raise RuntimeError(
            f"{dataset_name}: hanya ditemukan {len(selected)} QA unik "
            f"yang dapat dievaluasi; dibutuhkan {target_size}."
        )

    report = {
        "dataset": dataset_name,
        "source_total_records": len(data),
        "selected_count": len(selected),
        "skipped_non_qa_before_target_reached": skipped_non_qa,
        "skipped_duplicate_before_target_reached": skipped_duplicate,
        "first_selected": {
            "surah": selected[0]["surah"],
            "ayat": selected[0]["ayat"],
        },
        "last_selected": {
            "surah": selected[-1]["surah"],
            "ayat": selected[-1]["ayat"],
        },
        "unique_verse_count": len(seen),
    }

    return selected, report


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temporary.replace(path)


def main() -> None:
    configs = {
        "SPTP": {
            "input": SOURCE_DIR / "canonical_gold_qas_SPTP.json",
            "output": (
                OUTPUT_DIR
                / "canonical_gold_qas_gpt52_SPTP_first2600.json"
            ),
        },
        "MPTP": {
            "input": SOURCE_DIR / "canonical_gold_qas_MPTP.json",
            "output": (
                OUTPUT_DIR
                / "canonical_gold_qas_gpt52_MPTP_first2600.json"
            ),
        },
    }

    combined_report: dict[str, Any] = {
        "selection_rule": (
            "First 2,600 unique records with non-empty question and answer, "
            "selected independently for SPTP and MPTP."
        ),
        "target_size_per_dataset": TARGET_SIZE,
        "datasets": {},
    }

    selected_keys: dict[str, list[tuple[int, int]]] = {}

    for dataset_name, config in configs.items():
        source_data = load_json_list(config["input"])

        selected, report = select_first_evaluable(
            source_data,
            TARGET_SIZE,
            dataset_name,
        )

        save_json(config["output"], selected)
        combined_report["datasets"][dataset_name] = report
        selected_keys[dataset_name] = [
            verse_key(item) for item in selected
        ]

        print("=" * 72)
        print(f"{dataset_name}")
        print(f"Input       : {config['input']}")
        print(f"Output      : {config['output']}")
        print(f"Terpilih    : {len(selected):,}")
        print(f"Unik        : {report['unique_verse_count']:,}")
        print(f"Non-QA skip : {report['skipped_non_qa_before_target_reached']:,}")
        print(f"Duplikat    : {report['skipped_duplicate_before_target_reached']:,}")

    sptp_set = set(selected_keys["SPTP"])
    mptp_set = set(selected_keys["MPTP"])
    intersection = sptp_set & mptp_set

    combined_report["cross_dataset_comparison"] = {
        "same_order": selected_keys["SPTP"] == selected_keys["MPTP"],
        "same_verse_set": sptp_set == mptp_set,
        "intersection_size": len(intersection),
        "sptp_only_count": len(sptp_set - mptp_set),
        "mptp_only_count": len(mptp_set - sptp_set),
    }

    save_json(
        OUTPUT_DIR / "first2600_selection_report.json",
        combined_report,
    )

    print("=" * 72)
    print("SELESAI")
    print(f"SPTP terpilih             : {len(selected_keys['SPTP']):,}")
    print(f"MPTP terpilih             : {len(selected_keys['MPTP']):,}")
    print(f"Irisan ayat SPTP–MPTP     : {len(intersection):,}")
    print(
        "Set ayat sama             : "
        f"{combined_report['cross_dataset_comparison']['same_verse_set']}"
    )
    print(
        "Urutan sama               : "
        f"{combined_report['cross_dataset_comparison']['same_order']}"
    )
    print(
        "\nCatatan: perbedaan set/urutan tidak menghalangi evaluasi terpisah. "
        "Namun, analisis paired SPTP-vs-MPTP harus memakai irisannya."
    )


if __name__ == "__main__":
    main()
