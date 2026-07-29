import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI
from math import isclose

# =====================================
# CONFIGURATION
# =====================================

# -------- Validator A (Claude via proxy or replace endpoint) --------
endpoint_A = "https://hpfq1-mig3fe7b-eastus2.cognitiveservices.azure.com/"
deployment_A = "Deepseek-V3.2"
api_key_A = "7GjmRVvd9TQhZEB6rc0lf00nWemDBjgVrIEYaxKryo4CmyEBNP4SJQQJ99BKACHYHv6XJ3w3AAAAACOGw0z2"

client_A = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint_A,
    api_key=api_key_A,
)

# -------- Validator B (Gemini or other) --------
endpoint_B = "https://hpfq1-mig3fe7b-eastus2.cognitiveservices.azure.com/"
deployment_B = "Mistral-Large-3"
api_key_B = "7GjmRVvd9TQhZEB6rc0lf00nWemDBjgVrIEYaxKryo4CmyEBNP4SJQQJ99BKACHYHv6XJ3w3AAAAACOGw0z2"

client_B = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint_B,
    api_key=api_key_B,
)

INPUT_GOLD = "canonical_gold_qas.json"
VERSE_FILE = "canonical_verses_with_theme.json"
LOG_FILE = "validation_log.jsonl"
OUTPUT_CONSENSUS = "gold_consensus_supported.json"

# ⚠️ IMPORTANT: keep low for S0 tier
MAX_WORKERS = 2
SAVE_INTERVAL = 50

# =====================================
# LOAD DATA
# =====================================

with open(INPUT_GOLD, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

with open(VERSE_FILE, "r", encoding="utf-8") as f:
    verse_data = json.load(f)

# Build verse lookup
verse_lookup = {
    (v["surah"], v["ayat"]): v["text_id"]
    for v in verse_data
}

# =====================================
# LOAD EXISTING LOG (FOR RESUME)
# =====================================

validated_keys = set()
results = []

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            key = (record["surah"], record["ayat"])
            validated_keys.add(key)
            results.append(record)

print(f"Already validated: {len(validated_keys)}")

# Filter remaining
remaining_data = [
    entry for entry in gold_data
    if (entry["surah"], entry["ayat"]) not in validated_keys
]

print(f"Remaining to validate: {len(remaining_data)}")

# =====================================
# PROMPT BUILDER
# =====================================

def build_messages(entry):
    verse_text = verse_lookup.get((entry["surah"], entry["ayat"]), "")

    system_prompt = """
You are verifying whether a question–answer pair is fully supported by a provided Quran verse.

Rules:
- The answer must be fully and explicitly supported by the verse text.
- No inference beyond what is clearly stated.
- No external interpretation.
- Return only one word:
  SUPPORTED
  or
  NOT_SUPPORTED
""".strip()

    user_prompt = f"""
Verse:
"{verse_text}"

Question:
"{entry['question']}"

Answer:
"{entry['answer']}"

Is the answer fully supported?
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# =====================================
# RETRY CALL
# =====================================

def call_with_retry(client, deployment, messages):
    for attempt in range(5):
        try:
            return client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_tokens=10
            )
        except Exception as e:
            if "429" in str(e):
                wait = 2 ** attempt
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise e
    return None

# =====================================
# VALIDATE FUNCTION
# =====================================

def validate(entry):
    messages = build_messages(entry)

    resp_A = call_with_retry(client_A, deployment_A, messages)
    resp_B = call_with_retry(client_B, deployment_B, messages)

    if resp_A is None or resp_B is None:
        return None

    verdict_A = resp_A.choices[0].message.content.strip()
    verdict_B = resp_B.choices[0].message.content.strip()

    time.sleep(0.2)

    return {
        "surah": entry["surah"],
        "ayat": entry["ayat"],
        "verdict_A": verdict_A,
        "verdict_B": verdict_B
    }

# =====================================
# RUN VALIDATION
# =====================================

start_time = time.time()
total = len(gold_data)
processed = len(validated_keys)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(validate, entry): entry for entry in remaining_data}

    for i, future in enumerate(as_completed(futures), 1):

        result = future.result()
        if result:
            results.append(result)

            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            processed += 1

        if processed % SAVE_INTERVAL == 0:

            elapsed = time.time() - start_time
            avg = elapsed / (processed - len(validated_keys) + 1)
            remaining = total - processed
            eta = avg * remaining

            print("\n========== PROGRESS ==========")
            print(f"Processed: {processed}/{total}")
            print(f"Elapsed: {elapsed/60:.2f} min")
            print(f"ETA: {eta/60:.2f} min")
            print("================================")

# =====================================
# FINAL AGREEMENT CALCULATION
# =====================================

TP = TN = FP = FN = 0

for r in results:
    vA = r["verdict_A"]
    vB = r["verdict_B"]

    if vA == "SUPPORTED" and vB == "SUPPORTED":
        TP += 1
    elif vA == "NOT_SUPPORTED" and vB == "NOT_SUPPORTED":
        TN += 1
    elif vA == "SUPPORTED" and vB == "NOT_SUPPORTED":
        FP += 1
    elif vA == "NOT_SUPPORTED" and vB == "SUPPORTED":
        FN += 1

observed = (TP + TN) / len(results)

pA_sup = sum(1 for r in results if r["verdict_A"] == "SUPPORTED") / len(results)
pB_sup = sum(1 for r in results if r["verdict_B"] == "SUPPORTED") / len(results)

expected = (pA_sup * pB_sup) + ((1 - pA_sup) * (1 - pB_sup))
kappa = (observed - expected) / (1 - expected)

consensus_supported = [
    entry for entry in gold_data
    if any(
        r["surah"] == entry["surah"] and
        r["ayat"] == entry["ayat"] and
        r["verdict_A"] == "SUPPORTED" and
        r["verdict_B"] == "SUPPORTED"
        for r in results
    )
]

with open(OUTPUT_CONSENSUS, "w", encoding="utf-8") as f:
    json.dump(consensus_supported, f, ensure_ascii=False, indent=2)

total_time = time.time() - start_time

print("\n========== FINAL REPORT ==========")
print(f"Total validated: {len(results)}")
print(f"Observed Agreement: {observed:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Consensus Size: {len(consensus_supported)}")
print(f"Runtime: {total_time/60:.2f} min")
print("===================================")