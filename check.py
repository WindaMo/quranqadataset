import json

with open("canonical_verses_with_theme.json", "r", encoding="utf-8") as f:
    all_verses = json.load(f)

with open("canonical_gold_qas.json", "r", encoding="utf-8") as f:
    generated = json.load(f)

generated_keys = set((r["surah"], r["ayat"]) for r in generated)
all_keys = set((v["surah"], v["ayat"]) for v in all_verses)

missing = all_keys - generated_keys

print("Total expected:", len(all_keys))
print("Total generated:", len(generated_keys))
print("Missing verses:", len(missing))
print("Missing list:", list(missing)[:20])