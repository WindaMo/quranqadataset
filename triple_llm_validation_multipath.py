import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI
from itertools import combinations

# =====================================
# CONFIG
# =====================================
# ---- Validator A (DeepSeek) ----
client_A = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://hpfq1-mig3fe7b-eastus2.cognitiveservices.azure.com/",
    api_key="",
)
deployment_A = "DeepSeek-V3.2"

# ---- Validator B (Mistral) ----
client_B = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://hpfq1-mig3fe7b-eastus2.cognitiveservices.azure.com/",
    api_key="",
)
deployment_B = "Mistral-Large-3"

# ---- Validator C (Llama-4-Maverick-17B-128E-Instruct-FP8) ----
client_C = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://hpfq1-mig3fe7b-eastus2.cognitiveservices.azure.com/",
    api_key="",
)
deployment_C = "Llama-4-Maverick-17B-128E-Instruct-FP8"

INPUT_GOLD = "canonical_multipath_gold_qas.json"
VERSE_FILE = "canonical_verses_with_theme.json"

LOG_FILE = "validation_log_3models_multipath.jsonl"
OUTPUT_MAJORITY = "gold_majority_supported_multipath.json"
OUTPUT_STRICT = "gold_strict_supported_multipath.json"

MAX_WORKERS = 3
SAVE_INTERVAL = 50

# =====================================
# LOAD DATA
# =====================================

with open(INPUT_GOLD, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

for i, entry in enumerate(gold_data):
    if 'question' not in entry or 'answer' not in entry:
        print("Malformed entry at index:", i)
        print(entry)
        break

with open(VERSE_FILE, "r", encoding="utf-8") as f:
    verse_data = json.load(f)

verse_lookup = {}
for v in verse_data:
    key = (v.get("surah"), v.get("ayat"))
    if key[0] is not None and key[1] is not None:
        verse_lookup[key] = v.get("text_id", "")

# =====================================
# LOAD RESUME LOG
# =====================================

validated = {}
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                key = (record["surah"], record["ayat"])
                validated[key] = record
            except:
                print("Skipping corrupted log line.")
                continue

print("Already validated:", len(validated))

remaining_data = []

for entry in gold_data:
    if "question" not in entry or "answer" not in entry:
        print("Skipping non-QA entry:", entry)
        continue
    if (entry["surah"], entry["ayat"]) not in validated:
        remaining_data.append(entry)

print("Remaining:", len(remaining_data))

# =====================================
# PROMPT
# =====================================


def build_messages(entry):
    verse_text = verse_lookup.get((entry["surah"], entry["ayat"]), "")

    system_prompt = """
You are a strict binary classifier for textual grounding.

Task:
Determine whether the provided Answer is fully supported by the provided Quran verse.

Grounding Rules:
- The answer must be fully and explicitly supported by the verse text.
- No inference beyond what is clearly stated.
- No external interpretation.
- If any part of the answer goes beyond the verse text, it is NOT_SUPPORTED.

Output Rules:
- Do NOT explain.
- Do NOT analyze.
- Do NOT restate the verse.
- Output exactly one word:

SUPPORTED
or
NOT_SUPPORTED
""".strip()

    user_prompt = f"""
Verse:
{verse_text}

Question:
{entry['question']}

Answer:
{entry['answer']}

Output only SUPPORTED or NOT_SUPPORTED.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# =====================================
# RETRY WRAPPER
# =====================================


def call_with_retry(client, deployment, messages):
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_tokens=5,
                temperature=0
            )
            return response
        except Exception as e:
            if "429" in str(e):
                wait = 2 ** attempt
                print(f"Rate limit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print("ERROR:", e)
                return None
    return None

# =====================================
# VALIDATION FUNCTION
# =====================================


'''
print("\n=== DEBUG SINGLE CALL TEST ===")

test_entry = gold_data[0]
messages = build_messages(test_entry)

print("Testing Surah:", test_entry["surah"], "Ayat:", test_entry["ayat"])

response = call_with_retry(client_C, deployment_C, messages)

print("RAW RESPONSE:")
print(response)

if response:
    try:
        print("MESSAGE CONTENT:")
        print(response.choices[0].message.content)
        print("FINISH REASON:", response.choices[0].finish_reason)
    except Exception as e:
        print("Error accessing content:", e)

print("=== END DEBUG ===\n")

exit()
'''


def extract_verdict(response):
    if not response:
        return "ERROR"

    try:
        msg = response.choices[0].message

        # Case 1: Plain string
        if isinstance(msg.content, str):
            text = msg.content.strip()

        # Case 2: Structured content (list)
        elif isinstance(msg.content, list):
            text = ""
            for block in msg.content:
                if isinstance(block, dict) and "text" in block:
                    text += block["text"]
            text = text.strip()

        else:
            return "EMPTY"

        text_upper = text.upper()

        if "NOT_SUPPORTED" in text_upper:
            return "NOT_SUPPORTED"
        elif "SUPPORTED" in text_upper:
            return "SUPPORTED"
        else:
            return "EMPTY"

    except Exception as e:
        print("Parse error:", e)
        return "EMPTY"


def validate(entry):
    print(f"Start validating {entry['surah']}:{entry['ayat']}")
    messages = build_messages(entry)

    rA = call_with_retry(client_A, deployment_A, messages)
    rB = call_with_retry(client_B, deployment_B, messages)
    rC = call_with_retry(client_C, deployment_C, messages)

    if not rA or not rB or not rC:
        return None

    verdict_A = extract_verdict(rA)
    verdict_B = extract_verdict(rB)
    verdict_C = extract_verdict(rC)

    time.sleep(0.2)

    return {
        "surah": entry["surah"],
        "ayat": entry["ayat"],
        "A": verdict_A,
        "B": verdict_B,
        "C": verdict_C
    }

# =====================================
# RUN VALIDATION
# =====================================


start_time = time.time()
processed = len(validated)
total = len(gold_data)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(validate, e): e for e in remaining_data}

    for future in as_completed(futures):
        result = future.result()
        if result:
            key = (result["surah"], result["ayat"])
            validated[key] = result

            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            processed += 1

        if processed % SAVE_INTERVAL == 0:
            elapsed = time.time() - start_time
            avg = elapsed / max(1, processed)
            eta = avg * (total - processed)

            print("\n========== PROGRESS ==========")
            print(f"Processed: {processed}/{total}")
            print(f"Elapsed: {elapsed/60:.2f} min")
            print(f"ETA: {eta/60:.2f} min")
            print("================================")

# =====================================
# METRIC CALCULATION
# =====================================

records = list(validated.values())


def cohen_kappa(r1, r2):
    TP = TN = FP = FN = 0
    for r in records:
        v1 = r[r1]
        v2 = r[r2]
        if v1 == "SUPPORTED" and v2 == "SUPPORTED":
            TP += 1
        elif v1 == "NOT_SUPPORTED" and v2 == "NOT_SUPPORTED":
            TN += 1
        elif v1 == "SUPPORTED" and v2 == "NOT_SUPPORTED":
            FP += 1
        elif v1 == "NOT_SUPPORTED" and v2 == "SUPPORTED":
            FN += 1

    total = TP + TN + FP + FN
    observed = (TP + TN) / total
    p1_sup = sum(1 for r in records if r[r1] == "SUPPORTED") / total
    p2_sup = sum(1 for r in records if r[r2] == "SUPPORTED") / total
    expected = (p1_sup * p2_sup) + ((1 - p1_sup) * (1 - p2_sup))
    return observed, (observed - expected) / (1 - expected)


# Pairwise κ
pairs = list(combinations(["A", "B", "C"], 2))
for p in pairs:
    obs, k = cohen_kappa(p[0], p[1])
    print(f"Cohen κ {p[0]} vs {p[1]}: {k:.4f} (Agreement={obs:.4f})")

# =====================================
# FLEISS' KAPPA
# =====================================

N = len(records)
n = 3  # raters
category_counts = []

for r in records:
    sup = sum(1 for k in ["A", "B", "C"] if r[k] == "SUPPORTED")
    not_sup = 3 - sup
    category_counts.append([sup, not_sup])

p = [sum(c[i] for c in category_counts)/(N*n) for i in range(2)]
P_i = [(sum(c[j]**2 for j in range(2)) - n)/(n*(n-1)) for c in category_counts]
P_bar = sum(P_i)/N
P_e = sum(p_i**2 for p_i in p)
fleiss_kappa = (P_bar - P_e)/(1 - P_e)

print(f"\nFleiss' Kappa (3 raters): {fleiss_kappa:.4f}")

# =====================================
# EXPORT CONSENSUS DATASETS
# =====================================

majority = []
strict = []

for entry in gold_data:
    key = (entry["surah"], entry["ayat"])
    r = validated.get(key)
    if not r:
        continue

    votes = [r["A"], r["B"], r["C"]]
    sup_count = votes.count("SUPPORTED")

    if sup_count >= 2:
        majority.append(entry)

    if sup_count == 3:
        strict.append(entry)

with open(OUTPUT_MAJORITY, "w", encoding="utf-8") as f:
    json.dump(majority, f, ensure_ascii=False, indent=2)

with open(OUTPUT_STRICT, "w", encoding="utf-8") as f:
    json.dump(strict, f, ensure_ascii=False, indent=2)

print("\n========== FINAL REPORT ==========")
print(f"Total validated: {N}")
print(f"Majority consensus size: {len(majority)}")
print(f"Strict consensus size: {len(strict)}")
print("===================================")
