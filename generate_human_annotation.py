import json
import pandas as pd

# === FILE PATHS (adjust if needed) ===
MULTIPATH_LOG = "validation_log_3models_multipath.jsonl"
MULTIPATH_GOLD = "canonical_multipath_gold_qas.json"
CANONICAL_VERSES = "canonical_verses_with_theme.json"
OUTPUT_EXCEL = "human_annotation_multipath_non_strict.xlsx"

# --------------------------------------------------
# 1. Load validation log
# --------------------------------------------------
log_rows = []
with open(MULTIPATH_LOG, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            log_rows.append(json.loads(line))
        except:
            continue

log_df = pd.DataFrame(log_rows)

# Keep only fully validated rows
log_df = log_df[(log_df["A"] != "") & (log_df["B"] != "") & (log_df["C"] != "")]

# Identify non-strict cases
def is_strict_supported(row):
    return row["A"] == row["B"] == row["C"] == "SUPPORTED"

log_df["strict_supported"] = log_df.apply(is_strict_supported, axis=1)
non_strict_df = log_df[~log_df["strict_supported"]].copy()

print("Total multipath validated:", len(log_df))
print("Non-strict cases:", len(non_strict_df))

# --------------------------------------------------
# 2. Load multipath gold dataset
# --------------------------------------------------
with open(MULTIPATH_GOLD, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

gold_lookup = {
    (item["surah"], item["ayat"]): item
    for item in gold_data
    if "question" in item
}

# --------------------------------------------------
# 3. Load canonical verses (for text_id lookup)
# --------------------------------------------------
with open(CANONICAL_VERSES, "r", encoding="utf-8") as f:
    canonical_data = json.load(f)

verse_text_lookup = {
    (item["surah"], item["ayat"]): item.get("text_id", "")
    for item in canonical_data
}

# --------------------------------------------------
# 4. Build annotation rows
# --------------------------------------------------
rows = []

for _, row in non_strict_df.iterrows():
    key = (row["surah"], row["ayat"])
    gold_item = gold_lookup.get(key)
    verse_text = verse_text_lookup.get(key, "")

    if not gold_item:
        continue

    rows.append({
        "ID": f"{gold_item['surah']}:{gold_item['ayat']}",
        "Surah": gold_item["surah"],
        "Ayat": gold_item["ayat"],
        "Verse_Text": verse_text,  # FIXED HERE
        "Question": gold_item["question"],
        "Answer": gold_item["answer"],
        "Difficulty_Model": gold_item.get("difficulty", ""),
        "Theme_Paths": " || ".join(
            [" → ".join(path) for path in gold_item.get("theme_paths", [])]
        ),
        "Human_Verdict": "",
        "Error_Type": "",
        "Notes": ""
    })

annotation_df = pd.DataFrame(rows)

# --------------------------------------------------
# 5. Export Excel
# --------------------------------------------------
annotation_df.to_excel(OUTPUT_EXCEL, index=False)

print("\nExcel generated:", OUTPUT_EXCEL)
print("Total rows written:", len(annotation_df))