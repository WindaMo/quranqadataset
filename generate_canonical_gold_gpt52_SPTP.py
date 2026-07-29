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
INPUT_FILE = "canonical_verses_for_gold.json"
OUTPUT_FILE = "canonical_gold_qas.json"

# =====================================
# LOAD INPUT
# =====================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    verses = json.load(f)

# Resume capability
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = []

processed = set((r["surah"], r["ayat"]) for r in results)

print(f"Total verses: {len(verses)}")
print(f"Already processed: {len(processed)}")

# =====================================
# PROMPT BUILDER
# =====================================
def build_messages(surah, ayat, text):

    system_prompt = """
You are generating gold-standard evaluation data for a Quranic retrieval-augmented generation system.

General Rules:
- Use only the provided verse text.
- Do not use external knowledge.
- Do not use information from other verses.
- The question must be answerable strictly from this verse alone.
- The answer must be explicitly grounded in the verse text.
- Avoid trivial copy-paste questions.
- Avoid yes/no questions unless absolutely necessary.
- If the verse does not contain sufficient information to form a clear factual question, return exactly: INSUFFICIENT_EVIDENCE
- Return strictly valid JSON only.

Difficulty Classification Rules:
Classify difficulty based on structural and linguistic complexity of the verse, NOT on theological importance.

- easy:
  The answer is directly and clearly stated.
  The verse is short.
  Contains a single main concept.
  No numerical, legal, or conditional structure.

- medium:
  The verse contains multiple entities, descriptions, or related concepts.
  Requires selecting the correct portion of the verse.
  May include descriptive or explanatory structure.

- hard:
  The verse is long, legal, numerical, conditional, metaphorical, or conceptually dense.
  Requires careful parsing of structure or multiple components within the verse.

Base difficulty primarily on structural complexity of the verse, not on perceived religious significance.
""".strip()

    user_prompt = f"""
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
  "difficulty": "easy"
}}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# =====================================
# GENERATION FUNCTION
# =====================================
def generate_gold(entry, max_retries=3):

    surah = entry["surah"]
    ayat = entry["ayat"]
    text = entry["text_id"]

    if (surah, ayat) in processed:
        return None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=build_messages(surah, ayat, text),
                max_completion_tokens=400,
                response_format={"type": "json_object"}  # force JSON
            )

            content = response.choices[0].message.content.strip()

            if content == "INSUFFICIENT_EVIDENCE":
                return {
                    "surah": surah,
                    "ayat": ayat,
                    "status": "INSUFFICIENT_EVIDENCE"
                }

            parsed = json.loads(content)

            parsed["surah"] = surah
            parsed["ayat"] = ayat
            parsed["usage"] = response.usage.total_tokens

            return parsed

        except Exception as e:
            print(f"[Retry {attempt+1}] Error on {surah}:{ayat} -> {e}")
            time.sleep(1)

    return None

# =====================================
# PARALLEL EXECUTION
# =====================================
MAX_WORKERS = 6  # Safe start for GPT-5.2

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = {executor.submit(generate_gold, v): v for v in verses}

    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()

        if result:
            results.append(result)

        # Save every 25 results
        if i % 25 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"Saved {len(results)} results")

# Final save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Canonical gold generation completed.")
