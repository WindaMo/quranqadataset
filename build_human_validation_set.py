import json

DISAGREEMENT_FILE = "disagreement_cases.json"
MAJORITY_FILE = "gold_majority_supported.json"
VERSE_FILE = "canonical_verses_with_theme.json"

OUTPUT_FILE = "human_validation_set.json"

# Load files
with open(DISAGREEMENT_FILE, "r", encoding="utf-8") as f:
    disagreement_cases = json.load(f)

with open(MAJORITY_FILE, "r", encoding="utf-8") as f:
    majority_data = json.load(f)

with open(VERSE_FILE, "r", encoding="utf-8") as f:
    verse_data = json.load(f)

# Build key set
disagreement_keys = set(
    (item["surah"], item["ayat"])
    for item in disagreement_cases
)

# Build verse lookup with full theme paths
verse_lookup = {
    (v["surah"], v["ayat"]): {
        "text_id": v.get("text_id"),
        "theme_paths": v.get("theme_paths")  # full hierarchy list
    }
    for v in verse_data
}

human_validation_list = []

for entry in majority_data:

    key = (entry["surah"], entry["ayat"])

    if key in disagreement_keys:

        verse_info = verse_lookup.get(key, {})

        human_validation_list.append({
            "question": entry.get("question"),
            "answer": entry.get("answer"),
            "difficulty": entry.get("difficulty"),
            "surah": entry.get("surah"),
            "ayat": entry.get("ayat"),
            "text_id": verse_info.get("text_id", ""),
            "theme_path_used_for_generation": entry.get("theme_path_used"),
            "theme_paths_full": verse_info.get("theme_paths", [])  # ALL paths
        })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(human_validation_list, f, ensure_ascii=False, indent=2)

print("Total extracted for human validation:", len(human_validation_list))
print("Saved to:", OUTPUT_FILE)