import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from threading import Lock

from openai import OpenAI


BASE_URL = os.getenv("SILAB_BASE_URL", "https://llms.si-lab.org/api")
API_KEY = os.getenv("SILAB_API_KEY")
if not API_KEY:
    raise RuntimeError("SILAB_API_KEY belum diatur.")

DEFAULT_JUDGE_MODELS = ["gemma4:31b", "granite4.1:30b", "qwen3.6:35b"]
JUDGE_MODELS = [
    model.strip()
    for model in os.getenv("SILAB_JUDGE_MODELS", ",".join(DEFAULT_JUDGE_MODELS)).split(",")
    if model.strip()
]

if len(JUDGE_MODELS) != 3:
    raise ValueError("SILAB_JUDGE_MODELS must contain exactly 3 comma-separated model names.")

NUM_WORKERS = int(os.getenv("SILAB_JUDGE_NUM_WORKERS", os.getenv("SILAB_NUM_WORKERS", "3")))
SAVE_INTERVAL = int(os.getenv("SILAB_JUDGE_SAVE_INTERVAL", "50"))
MAX_RETRY_DELAY_SECONDS = 120
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_KEYWORDS = (
    "connection",
    "connect",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "rate limit",
    "too many requests",
    "server error",
    "bad gateway",
    "gateway timeout",
    "reset by peer",
)

LOCAL_TZ = timezone(timedelta(hours=7), "Asia/Jakarta")
CATEGORIES = {"SUPPORTED", "NOT_SUPPORTED"}

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

log_lock = Lock()


def project_root():
    return Path(__file__).resolve().parents[2]


def model_slug(model_name):
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def save_json_atomic(path, data):
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(temp_path, path)


def append_jsonl(path, record):
    with log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def timestamp_fields():
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now(LOCAL_TZ)
    return {
        "timestamp": now_utc.isoformat(),
        "timestamp_local": now_local.isoformat(),
    }


def retry_delay_seconds(attempt):
    return min(MAX_RETRY_DELAY_SECONDS, 2 ** min(attempt, 7))


def is_retryable_error(error):
    status_code = getattr(error, "status_code", None)

    if status_code in RETRYABLE_STATUS_CODES:
        return True

    message = str(error).lower()
    return any(keyword in message for keyword in RETRYABLE_ERROR_KEYWORDS)


def build_progress_bar(completed, total, width=32):
    if total <= 0:
        return "[" + ("#" * width) + "] 100.00%"

    fraction = min(max(completed / total, 0), 1)
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {fraction * 100:6.2f}%"


def progress_snapshot(start_time, completed, total):
    elapsed = time.time() - start_time
    avg_time = elapsed / completed if completed > 0 else 0
    remaining = max(total - completed, 0)
    eta_seconds = avg_time * remaining if completed > 0 else 0
    estimated_finish_local = datetime.now(LOCAL_TZ) + timedelta(seconds=eta_seconds)

    return {
        "elapsed": elapsed,
        "avg_time": avg_time,
        "remaining": remaining,
        "eta_seconds": eta_seconds,
        "estimated_finish_local": estimated_finish_local,
    }


def build_messages(entry, verse_lookup):
    verse_text = verse_lookup.get((entry["surah"], entry["ayat"]), entry.get("text_id", ""))

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
        {"role": "user", "content": user_prompt},
    ]


def call_with_retry(model, messages, retry_log_path, entry, start_time):
    attempt = 0

    while True:
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=8,
                temperature=0,
            )
        except Exception as e:
            if not is_retryable_error(e):
                append_jsonl(
                    retry_log_path,
                    {
                        **timestamp_fields(),
                        "event": "judge_error",
                        "surah": entry["surah"],
                        "ayat": entry["ayat"],
                        "judge_model": model,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "elapsed_seconds": time.time() - start_time,
                    },
                )
                return None

            attempt += 1
            wait_seconds = retry_delay_seconds(attempt)

            append_jsonl(
                retry_log_path,
                {
                    **timestamp_fields(),
                    "event": "retry_waiting",
                    "surah": entry["surah"],
                    "ayat": entry["ayat"],
                    "judge_model": model,
                    "attempt": attempt,
                    "wait_seconds": wait_seconds,
                    "elapsed_seconds": time.time() - start_time,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )

            print(
                f"\nRetry judge {model} untuk {entry['surah']}:{entry['ayat']} "
                f"(attempt {attempt}, tunggu {format_duration(wait_seconds)})"
            )
            time.sleep(wait_seconds)


def extract_verdict(response):
    if not response:
        return "ERROR"

    try:
        msg = response.choices[0].message

        if isinstance(msg.content, str):
            text = msg.content.strip()
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
        if "SUPPORTED" in text_upper:
            return "SUPPORTED"
        return "EMPTY"

    except Exception:
        return "EMPTY"


def validate_entry(entry, verse_lookup, log_file, start_time):
    item_start_time = time.time()
    messages = build_messages(entry, verse_lookup)
    verdicts = {}
    token_usage = {}

    for label, model in zip(["A", "B", "C"], JUDGE_MODELS):
        response = call_with_retry(model, messages, log_file, entry, start_time)
        verdicts[label] = extract_verdict(response)
        token_usage[label] = response.usage.total_tokens if response and response.usage else None

    return {
        **timestamp_fields(),
        "event": "validated",
        "surah": entry["surah"],
        "ayat": entry["ayat"],
        "A": verdicts["A"],
        "B": verdicts["B"],
        "C": verdicts["C"],
        "model_A": JUDGE_MODELS[0],
        "model_B": JUDGE_MODELS[1],
        "model_C": JUDGE_MODELS[2],
        "tokens_A": token_usage["A"],
        "tokens_B": token_usage["B"],
        "tokens_C": token_usage["C"],
        "duration_seconds": time.time() - item_start_time,
        "elapsed_seconds": time.time() - start_time,
    }


def load_validated(log_file):
    validated = {}

    if not log_file.exists():
        return validated

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print("Skipping corrupted log line.")
                continue

            if record.get("event") != "validated":
                continue

            # Hanya tiga verdict sah yang dianggap selesai.
            if not all(
                record.get(label) in CATEGORIES
                for label in ["A", "B", "C"]
            ):
                continue

            key = (record["surah"], record["ayat"])
            validated[key] = record

    return validated


def load_verse_lookup(verse_file):
    with open(verse_file, "r", encoding="utf-8") as f:
        verse_data = json.load(f)

    verse_lookup = {}
    for verse in verse_data:
        key = (verse.get("surah"), verse.get("ayat"))
        if key[0] is not None and key[1] is not None:
            verse_lookup[key] = verse.get("text_id", "")

    return verse_lookup


def cohen_kappa(records, r1, r2):
    valid = [
        record for record in records
        if record.get(r1) in CATEGORIES and record.get(r2) in CATEGORIES
    ]

    if not valid:
        return 0, 0

    tp = tn = fp = fn = 0
    for record in valid:
        v1 = record[r1]
        v2 = record[r2]
        if v1 == "SUPPORTED" and v2 == "SUPPORTED":
            tp += 1
        elif v1 == "NOT_SUPPORTED" and v2 == "NOT_SUPPORTED":
            tn += 1
        elif v1 == "SUPPORTED" and v2 == "NOT_SUPPORTED":
            fp += 1
        elif v1 == "NOT_SUPPORTED" and v2 == "SUPPORTED":
            fn += 1

    total = tp + tn + fp + fn
    if total == 0:
        return 0, 0

    observed = (tp + tn) / total
    p1_sup = sum(1 for record in valid if record[r1] == "SUPPORTED") / total
    p2_sup = sum(1 for record in valid if record[r2] == "SUPPORTED") / total
    expected = (p1_sup * p2_sup) + ((1 - p1_sup) * (1 - p2_sup))

    if expected == 1:
        return observed, 1 if observed == 1 else 0

    return observed, (observed - expected) / (1 - expected)


def fleiss_kappa(records):
    valid = [
        record for record in records
        if all(record.get(label) in CATEGORIES for label in ["A", "B", "C"])
    ]

    if not valid:
        return 0, 0

    n_records = len(valid)
    n_raters = 3
    category_counts = []

    for record in valid:
        supported = sum(1 for label in ["A", "B", "C"] if record[label] == "SUPPORTED")
        not_supported = n_raters - supported
        category_counts.append([supported, not_supported])

    p = [
        sum(counts[i] for counts in category_counts) / (n_records * n_raters)
        for i in range(2)
    ]
    p_i = [
        (sum(counts[j] ** 2 for j in range(2)) - n_raters) / (n_raters * (n_raters - 1))
        for counts in category_counts
    ]
    p_bar = sum(p_i) / n_records
    p_e = sum(value ** 2 for value in p)

    if p_e == 1:
        return (1 if p_bar == 1 else 0), n_records

    return (p_bar - p_e) / (1 - p_e), n_records


def build_consensus(gold_data, validated):
    majority = []
    strict = []

    for entry in gold_data:
        if "question" not in entry or "answer" not in entry:
            continue

        key = (entry["surah"], entry["ayat"])
        record = validated.get(key)
        if not record:
            continue

        votes = [record["A"], record["B"], record["C"]]
        supported_count = votes.count("SUPPORTED")

        if supported_count >= 2:
            majority.append(entry)

        if supported_count == 3:
            strict.append(entry)

    return majority, strict


def run_validation(dataset_type, input_gold, generator_model_name, verse_file=None):
    root = project_root()
    script_dir = Path(__file__).resolve().parent
    input_gold = Path(input_gold)
    verse_file = Path(verse_file) if verse_file else root / "canonical_verses_with_theme.json"

    if not input_gold.exists():
        raise FileNotFoundError(f"Input gold file not found: {input_gold}")

    if not verse_file.exists():
        raise FileNotFoundError(f"Verse file not found: {verse_file}")

    generator_slug = model_slug(generator_model_name)
    dataset_slug = dataset_type.upper()

    log_file = Path(os.getenv(
        f"SILAB_{dataset_slug}_JUDGE_LOG",
        script_dir / f"validation_log_3models_{generator_slug}_{dataset_slug}.jsonl",
    ))
    output_majority = Path(os.getenv(
        f"SILAB_{dataset_slug}_OUTPUT_MAJORITY",
        script_dir / f"gold_majority_supported_{generator_slug}_{dataset_slug}.json",
    ))
    output_strict = Path(os.getenv(
        f"SILAB_{dataset_slug}_OUTPUT_STRICT",
        script_dir / f"gold_strict_supported_{generator_slug}_{dataset_slug}.json",
    ))
    summary_file = Path(os.getenv(
        f"SILAB_{dataset_slug}_JUDGE_SUMMARY",
        script_dir / f"validation_summary_{generator_slug}_{dataset_slug}.json",
    ))

    with open(input_gold, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    verse_lookup = load_verse_lookup(verse_file)
    validated = load_validated(log_file)

    remaining_data = []
    skipped_non_qa = 0

    for entry in gold_data:
        if "question" not in entry or "answer" not in entry:
            skipped_non_qa += 1
            continue

        key = (entry["surah"], entry["ayat"])
        if key not in validated:
            remaining_data.append(entry)

    print(f"Dataset type: {dataset_slug}")
    print(f"Generator model: {generator_model_name}")
    print(f"Input gold: {input_gold}")
    print(f"Verse file: {verse_file}")
    print(f"Log file: {log_file}")
    print(f"Majority output: {output_majority}")
    print(f"Strict output: {output_strict}")
    print(f"Judge models: {', '.join(JUDGE_MODELS)}")
    print(f"Parallel workers: {NUM_WORKERS}")
    print(f"Total input entries: {len(gold_data)}")
    print(f"Skipped non-QA entries: {skipped_non_qa}")
    print(f"Already validated: {len(validated)}")
    print(f"Remaining: {len(remaining_data)}")

    append_jsonl(
        log_file,
        {
            **timestamp_fields(),
            "event": "run_started",
            "dataset_type": dataset_slug,
            "generator_model": generator_model_name,
            "input_gold": str(input_gold),
            "verse_file": str(verse_file),
            "judge_models": JUDGE_MODELS,
            "parallel_workers": NUM_WORKERS,
            "total_input_entries": len(gold_data),
            "skipped_non_qa_entries": skipped_non_qa,
            "already_validated": len(validated),
            "remaining": len(remaining_data),
        },
    )

    start_time = time.time()
    completed_this_run = 0
    total_to_run = len(remaining_data)

    if total_to_run == 0:
        print("No remaining entries to validate.")
    else:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(validate_entry, entry, verse_lookup, log_file, start_time): entry
                for entry in remaining_data
            }

            for future in as_completed(futures):
                result = future.result()
                key = (result["surah"], result["ayat"])
                validated[key] = result
                append_jsonl(log_file, result)

                completed_this_run += 1
                snapshot = progress_snapshot(start_time, completed_this_run, total_to_run)
                bar = build_progress_bar(completed_this_run, total_to_run)
                print(
                    f"\r{bar} {completed_this_run}/{total_to_run} "
                    f"| total validated {len(validated)} "
                    f"| ETA {format_duration(snapshot['eta_seconds'])} "
                    f"| finish {snapshot['estimated_finish_local'].strftime('%Y-%m-%d %H:%M:%S %Z')}",
                    end="",
                    flush=True,
                )

                if completed_this_run % SAVE_INTERVAL == 0 or completed_this_run == total_to_run:
                    majority, strict = build_consensus(gold_data, validated)
                    save_json_atomic(output_majority, majority)
                    save_json_atomic(output_strict, strict)

                    print("\n========== PROGRESS ==========")
                    print(f"Dataset type: {dataset_slug}")
                    print(f"Completed this run: {completed_this_run}/{total_to_run}")
                    print(f"Total validated: {len(validated)}")
                    print(f"Majority consensus size: {len(majority)}")
                    print(f"Strict consensus size: {len(strict)}")
                    print(f"Elapsed: {format_duration(snapshot['elapsed'])}")
                    print(f"Avg per item: {snapshot['avg_time']:.2f} sec")
                    print(f"ETA: {format_duration(snapshot['eta_seconds'])}")
                    print("================================")

                    append_jsonl(
                        log_file,
                        {
                            **timestamp_fields(),
                            "event": "progress_saved",
                            "dataset_type": dataset_slug,
                            "completed_this_run": completed_this_run,
                            "total_to_run": total_to_run,
                            "total_validated": len(validated),
                            "majority_consensus_size": len(majority),
                            "strict_consensus_size": len(strict),
                            "elapsed_seconds": snapshot["elapsed"],
                            "average_seconds_per_item": snapshot["avg_time"],
                            "eta_seconds": snapshot["eta_seconds"],
                            "estimated_finish_local": snapshot["estimated_finish_local"].isoformat(),
                        },
                    )

        print()

    records = list(validated.values())
    majority, strict = build_consensus(gold_data, validated)
    save_json_atomic(output_majority, majority)
    save_json_atomic(output_strict, strict)

    pairwise = {}
    for label_a, label_b in combinations(["A", "B", "C"], 2):
        observed, kappa = cohen_kappa(records, label_a, label_b)
        pairwise[f"{label_a}_vs_{label_b}"] = {
            "agreement": observed,
            "cohen_kappa": kappa,
        }

    fleiss, metric_record_count = fleiss_kappa(records)

    summary = {
        "dataset_type": dataset_slug,
        "generator_model": generator_model_name,
        "input_gold": str(input_gold),
        "verse_file": str(verse_file),
        "judge_models": {
            "A": JUDGE_MODELS[0],
            "B": JUDGE_MODELS[1],
            "C": JUDGE_MODELS[2],
        },
        "total_input_entries": len(gold_data),
        "skipped_non_qa_entries": skipped_non_qa,
        "total_validated": len(validated),
        "metric_record_count": metric_record_count,
        "majority_consensus_size": len(majority),
        "strict_consensus_size": len(strict),
        "pairwise": pairwise,
        "fleiss_kappa": fleiss,
        "majority_output": str(output_majority),
        "strict_output": str(output_strict),
        "log_file": str(log_file),
        "finished_at_local": datetime.now(LOCAL_TZ).isoformat(),
    }

    save_json_atomic(summary_file, summary)

    append_jsonl(
        log_file,
        {
            **timestamp_fields(),
            "event": "run_finished",
            **summary,
        },
    )

    print("\n========== FINAL REPORT ==========")
    print(f"Dataset type: {dataset_slug}")
    print(f"Total validated: {len(validated)}")
    print(f"Majority consensus size: {len(majority)}")
    print(f"Strict consensus size: {len(strict)}")
    for pair_name, values in pairwise.items():
        print(
            f"Cohen Kappa {pair_name}: {values['cohen_kappa']:.4f} "
            f"(Agreement={values['agreement']:.4f})"
        )
    print(f"Fleiss Kappa (3 raters): {fleiss:.4f}")
    print(f"Summary file: {summary_file}")
    print("===================================")
