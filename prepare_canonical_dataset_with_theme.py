import json

# =========================
# FILE PATHS
# =========================
TEMATIK_FILE = "tematik.json"
AYAT_FILE = "ayat.json"
OUTPUT_FILE = "canonical_verses_with_theme.json"

# =========================
# LOAD FILES
# =========================
with open(TEMATIK_FILE, "r", encoding="utf-8") as f:
    tematik_data = json.load(f)

with open(AYAT_FILE, "r", encoding="utf-8") as f:
    ayat_data = json.load(f)

# =========================
# STEP 1: EXTRACT VERSE → THEME PATHS
# =========================
# Mapping:
# (surah, ayat) → list of theme paths

verse_theme_map = {}

def traverse(node, path):

    if isinstance(node, dict):
        for key, value in node.items():
            traverse(value, path + [key])

    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict):
                if "surah" in item and "ayat" in item:
                    key = (item["surah"], item["ayat"])

                    if key not in verse_theme_map:
                        verse_theme_map[key] = []

                    verse_theme_map[key].append(path)
                else:
                    traverse(item, path)
            else:
                traverse(item, path)

# Run traversal
traverse(tematik_data, [])

print(f"Total unique verses found: {len(verse_theme_map)}")

# =========================
# STEP 2: BUILD AYAT LOOKUP
# =========================
ayat_lookup = {}

for verse in ayat_data:
    key = (verse["id_surah"], verse["ayat"])
    ayat_lookup[key] = verse["ayat_bahasa_indonesia"]

# =========================
# STEP 3: BUILD FINAL DATASET
# =========================
final_dataset = []
invalid_references = []

for key, theme_paths in verse_theme_map.items():
    if key in ayat_lookup:
        surah, ayat = key

        final_dataset.append({
            "surah": surah,
            "ayat": ayat,
            "text_id": ayat_lookup[key],
            "theme_paths": theme_paths  # list of full hierarchical paths
        })
    else:
        invalid_references.append(key)

print(f"Valid verses: {len(final_dataset)}")
print(f"Invalid thematic references removed: {len(invalid_references)}")

# =========================
# SAVE OUTPUT
# =========================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2)

print(f"Saved to: {OUTPUT_FILE}")