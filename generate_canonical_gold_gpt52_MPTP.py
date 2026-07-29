import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI

# =====================================
# AZURE CONFIG
# =====================================
endpoint = "https://hpfq1-mig3fe7b-eastus2.cognitiveservices.azure.com/"
deployment = "gpt-5.2-chat"
subscription_key = ""
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# =====================================
# FILE CONFIG
# =====================================
INPUT_FILE = "canonical_verses_with_theme.json"
OUTPUT_FILE = "canonical_multipath_gold_qas.json"
FILTERED_LOG = "filtered_verses_multipath.json"

# =====================================
# LOAD INPUT
# =====================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    verses = json.load(f)

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = []

if os.path.exists(FILTERED_LOG):
    with open(FILTERED_LOG, "r", encoding="utf-8") as f:
        filtered_log = json.load(f)
else:
    filtered_log = []

processed = set((r["surah"], r["ayat"]) for r in results)

print(f"Total verses: {len(verses)}")
print(f"Already processed: {len(processed)}")

# =====================================
# GLOBAL METRICS
# =====================================
start_time = time.time()
total_tokens_used = 0
insufficient_count = 0
filtered_count = 0

# =====================================
# PROMPT BUILDER
# =====================================
def build_messages(surah, ayat, text, theme_paths):

    # === FORMAT MULTIPLE THEME PATHS ===
    if theme_paths:
        formatted_paths = []
        for i, path in enumerate(theme_paths, 1):
            formatted_paths.append(f"{i}. " + " → ".join(path))
        theme_string = "\n".join(formatted_paths)
    else:
        theme_string = "None"

    system_prompt = """
This task processes Quranic scripture strictly for academic dataset construction and retrieval evaluation.

The content may contain references to warfare, punishment, divine judgment, or condemnation as part of religious scripture.
These references must be handled neutrally and descriptively.
The output must not promote, justify, or glorify violence.

The verse text is in Bahasa Indonesia.
The generated question and answer MUST be written strictly in Bahasa Indonesia.

Language Rules:
- The question MUST be entirely in Bahasa Indonesia.
- The answer MUST be entirely in Bahasa Indonesia.
- Do NOT use English words.
- Only JSON keys remain in English.

The verse belongs to multiple thematic categories.
These thematic paths provide contextual orientation only.
They must NOT introduce information beyond the verse text.

The question must simulate a natural user query.
The user does NOT see the verse.

Strict Rules:
- Do NOT refer to “ayat ini” or similar phrases.
- Do NOT assume user sees the verse.
- The answer must be supported strictly by the verse text.
- Do NOT add external information.
- If too fragmentary, return exactly: INSUFFICIENT_EVIDENCE.
- Return strictly valid JSON only.

Difficulty Classification:
- easy: short verse, single concept.
- medium: multiple concepts.
- hard: long, legal, numerical, conditional, metaphorical.
""".strip()

    user_prompt = f"""
Theme path: {theme_string}
Surah: {surah}
Ayat: {ayat}
Text: "{text}"

Generate one single-hop question and answer pair.

Return JSON:
{{
  "question": "...",
  "answer": "...",
  "supporting_verses": [
    {{"surah": {surah}, "ayat": {ayat}}}
  ],
  "type": "single-hop",
  "difficulty": "easy | medium | hard"
}}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# =====================================
# GENERATION FUNCTION
# =====================================
def generate_gold(entry):

    global total_tokens_used
    global insufficient_count
    global filtered_count

    surah = entry["surah"]
    ayat = entry["ayat"]
    text = entry["text_id"]
    theme_paths = entry.get("theme_paths", [])

    if (surah, ayat) in processed:
        return None

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=build_messages(surah, ayat, text, theme_paths),
            max_completion_tokens=400,
            response_format={"type": "json_object"}
        )

        total_tokens_used += response.usage.total_tokens

        content = response.choices[0].message.content.strip()

        if content == "INSUFFICIENT_EVIDENCE":
            insufficient_count += 1
            return {
                "surah": surah,
                "ayat": ayat,
                "status": "INSUFFICIENT_EVIDENCE"
            }

        parsed = json.loads(content)

        parsed["surah"] = surah
        parsed["ayat"] = ayat
        # Store ALL theme paths since all were provided in prompt
        parsed["theme_paths"] = theme_paths

        return parsed

    except Exception as e:
        if "content_filter" in str(e):
            filtered_count += 1
            filtered_log.append({"surah": surah, "ayat": ayat})
            return None

        print(f"Error on {surah}:{ayat} -> {e}")
        return None

# =====================================
# PARALLEL EXECUTION
# =====================================
MAX_WORKERS = 6

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = {executor.submit(generate_gold, v): v for v in verses}

    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()

        if result:
            results.append(result)

        # Save every 25
        if i % 25 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            with open(FILTERED_LOG, "w", encoding="utf-8") as f:
                json.dump(filtered_log, f, ensure_ascii=False, indent=2)

            elapsed = time.time() - start_time
            processed_count = len(results)
            total = len(verses)

            avg_time = elapsed / processed_count if processed_count > 0 else 0
            remaining = total - processed_count
            eta_seconds = avg_time * remaining

            print("\n========== PROGRESS ==========")
            print(f"Processed: {processed_count}/{total}")
            print(f"Elapsed: {elapsed/60:.2f} min")
            print(f"Avg per verse: {avg_time:.2f} sec")
            print(f"ETA: {eta_seconds/60:.2f} min")
            print(f"Tokens used: {total_tokens_used}")
            print(f"Filtered: {filtered_count}")
            print(f"Insufficient: {insufficient_count}")
            print("================================")

# =====================================
# FINAL SAVE & SUMMARY
# =====================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open(FILTERED_LOG, "w", encoding="utf-8") as f:
    json.dump(filtered_log, f, ensure_ascii=False, indent=2)

total_time = time.time() - start_time

print("\n========== FINAL SUMMARY ==========")
print(f"Total runtime: {total_time/60:.2f} minutes")
print(f"Total verses processed: {len(results)}")
print(f"Average time per verse: {total_time/len(results):.2f} sec")
print(f"Total tokens used: {total_tokens_used}")
print(f"Filtered verses: {filtered_count}")
print(f"Insufficient evidence: {insufficient_count}")
print("===================================")
