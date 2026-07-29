import json
import pandas as pd

INPUT_FILE = "human_validation_set.json"
OUTPUT_FILE = "human_annotation_sheet.xlsx"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for item in data:
    rows.append({
        "ID": f"{item['surah']}:{item['ayat']}",
        "Surah": item["surah"],
        "Ayat": item["ayat"],
        "Verse_Text": item["text_id"],
        "Question": item["question"],
        "Answer": item["answer"],
        "Difficulty_Model": item.get("difficulty", ""),
        "Theme_Paths": " || ".join(
            [" → ".join(path) for path in item.get("theme_paths_full", [])]
        ),
        "Human_Verdict": "",  # to fill
        "Error_Type": "",     # optional taxonomy
        "Notes": ""           # optional comments
    })

df = pd.DataFrame(rows)
df.to_excel(OUTPUT_FILE, index=False)

print("Annotation sheet saved to:", OUTPUT_FILE)
print("Total items:", len(df))