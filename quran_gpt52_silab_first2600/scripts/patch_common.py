#!/usr/bin/env python3
"""
Patch the copied SI-LAB judge helper:
1. Remove a hard-coded API-key fallback.
2. Treat ERROR/EMPTY verdict records as unfinished so they can be retried.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "silab" / "judges" / "silab_triple_judge_common.py"

if not TARGET.exists():
    raise FileNotFoundError(f"Tidak ditemukan: {TARGET}")

text = TARGET.read_text(encoding="utf-8")

# Replace API key fallback with environment-only configuration.
text, api_replacements = re.subn(
    r'API_KEY\s*=\s*os\.getenv\("SILAB_API_KEY",\s*"[^"]*"\)',
    'API_KEY = os.getenv("SILAB_API_KEY")\n'
    'if not API_KEY:\n'
    '    raise RuntimeError("SILAB_API_KEY belum diatur.")',
    text,
    count=1,
)

start = text.find("def load_validated(log_file):")
end = text.find("def load_verse_lookup(verse_file):")

if start == -1 or end == -1 or end <= start:
    raise RuntimeError(
        "Blok load_validated() tidak ditemukan pada common judge."
    )

replacement = """def load_validated(log_file):
    validated = {}

    if not log_file.exists():
        return validated

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print("Skipping corrupted log line.")
                continue

            if record.get("event") != "validated":
                continue

            # Hanya tiga verdict sah yang dianggap selesai.
            if not all(
                record.get(label) in CATEGORIES
                for label in ["A", "B", "C"]
            ):
                continue

            key = (record["surah"], record["ayat"])
            validated[key] = record

    return validated


"""

text = text[:start] + replacement + text[end:]
TARGET.write_text(text, encoding="utf-8")

if api_replacements == 0:
    print(
        "Peringatan: fallback API key tidak ditemukan atau sudah pernah dipatch."
    )

print(f"Patch selesai: {TARGET}")
