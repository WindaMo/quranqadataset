import json

INPUT_LOG = "validation_log_3models.jsonl"
OUTPUT_FILE = "disagreement_cases.json"

disagreements = []

with open(INPUT_LOG, "r", encoding="utf-8") as f:
    for line in f:
        try:
            record = json.loads(line)

            # Skip corrupted lines
            if not all(k in record for k in ["surah", "ayat", "A", "B", "C"]):
                continue

            votes = [record["A"], record["B"], record["C"]]

            # If at least one NOT_SUPPORTED
            if "NOT_SUPPORTED" in votes:
                disagreements.append({
                    "surah": record["surah"],
                    "ayat": record["ayat"]
                })

        except:
            continue

print("Total disagreement cases:", len(disagreements))

# Save to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(disagreements, f, ensure_ascii=False, indent=2)

print("Saved to:", OUTPUT_FILE)