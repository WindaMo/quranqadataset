import json
import os

# =========================
# FILE PATHS
# =========================
TEMATIK_FILE = "tematik.json"
AYAT_FILE = "ayat.json"
OUTPUT_FILE = "canonical_verses_for_gold.json"

# =========================
# STEP 1: LOAD FILES
# =========================
with open(TEMATIK_FILE, "r", encoding="utf-8") as f:
    tematik_data = json.load(f)

with open(AYAT_FILE, "r", encoding="utf-8") as f:
    ayat_data = json.load(f)

# =========================
# STEP 2: EXTRACT UNIQUE (SURAH, AYAT)
# =========================
unique_verses = set()

def extract_verses(node):
    """
    Recursively traverse tematik.json
    and extract all {surah, ayat} pairs.
    """
    if isinstance(node, dict):
        for value in node.values():
            extract_verses(value)

    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict):
                # Check if this is a verse reference
                if "surah" in item and "ayat" in item:
                    unique_verses.add((item["surah"], item["ayat"]))
                else:
                    extract_verses(item)
            else:
                extract_verses(item)

# Run extraction
extract_verses(tematik_data)

print(f"Total unique verses found in thematic structure: {len(unique_verses)}")

# =========================
# STEP 3: CREATE LOOKUP FROM AYAT.JSON
# =========================
ayat_lookup = {}

for verse in ayat_data:
    key = (verse["id_surah"], verse["ayat"])
    ayat_lookup[key] = verse["ayat_bahasa_indonesia"]

# =========================
# STEP 4: BUILD FINAL DATASET
# =========================
final_dataset = []
missing_verses = []

for surah, ayat in sorted(unique_verses):
    key = (surah, ayat)

    if key in ayat_lookup:
        final_dataset.append({
            "surah": surah,
            "ayat": ayat,
            "text_id": ayat_lookup[key]
        })
    else:
        missing_verses.append((surah, ayat))

print(f"Total matched verses: {len(final_dataset)}")
print(f"Missing verses count: {len(missing_verses)}")

if missing_verses:
    print("Missing verses:")
    for m in missing_verses:
        print(m)

# =========================
# STEP 5: SAVE OUTPUT
# =========================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2)

print(f"\nCanonical dataset saved to: {OUTPUT_FILE}")