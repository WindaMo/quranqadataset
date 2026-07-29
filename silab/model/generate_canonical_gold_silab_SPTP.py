import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

from openai import OpenAI

# =====================================
# SI-LAB CONFIG
# =====================================
BASE_URL = "https://llms.si-lab.org/api"
API_KEY = os.getenv("SILAB_API_KEY", "sk-1194efecbf0b45d6ac175dc9b06a428a")

# Ganti model di sini jika ingin menjalankan model lain, contoh: "gemma3:1b"
MODEL_NAME = "gpt-oss:120b"

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# Ubah angka ini untuk mengatur jumlah request paralel.
# Bisa juga dioverride dari terminal: $env:SILAB_NUM_WORKERS="10"
NUM_WORKERS = int(os.getenv("SILAB_NUM_WORKERS", "6"))
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

# =====================================
# FILE CONFIG
# =====================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_SLUG = re.sub(r"[^a-zA-Z0-9]+", "_", MODEL_NAME).strip("_").lower()

INPUT_FILE = PROJECT_ROOT / "canonical_verses_with_theme.json"
OUTPUT_FILE = SCRIPT_DIR / f"canonical_gold_qas_{MODEL_SLUG}_SPTP-v2.json"
FILTERED_LOG = SCRIPT_DIR / f"filtered_verses_{MODEL_SLUG}_SPTP-v2.json"
LOG_FILE = SCRIPT_DIR / f"generation_log_{MODEL_SLUG}_SPTP-v2.jsonl"

# =====================================
# LOAD INPUT
# =====================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    verses = json.load(f)

if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = []

legacy_results = [r for r in results if "theme_path_used" not in r]
if legacy_results:
    print(
        "Ignoring legacy SPTP results without theme_path_used: "
        f"{len(legacy_results)} entries"
    )
    results = [r for r in results if "theme_path_used" in r]

if FILTERED_LOG.exists():
    with open(FILTERED_LOG, "r", encoding="utf-8") as f:
        filtered_log = json.load(f)
else:
    filtered_log = []

processed = set((r["surah"], r["ayat"]) for r in results)
filtered_keys = set((r["surah"], r["ayat"]) for r in filtered_log)
pending_verses = [
    v for v in verses
    if (v["surah"], v["ayat"]) not in processed
    and (v["surah"], v["ayat"]) not in filtered_keys
]

print(f"Model: {MODEL_NAME}")
print(f"Input file: {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")
print(f"Filtered log: {FILTERED_LOG}")
print(f"Generation log: {LOG_FILE}")
print(f"Total verses: {len(verses)}")
print(f"Already processed: {len(processed)}")
print(f"Already filtered/skipped: {len(filtered_keys)}")
print(f"Remaining to process: {len(pending_verses)}")
print(f"Parallel workers: {NUM_WORKERS}")

# =====================================
# GLOBAL METRICS
# =====================================
start_time = time.time()
initial_results_count = len(results)
total_tokens_used = 0
insufficient_count = 0
filtered_count = 0
parse_error_count = 0
log_lock = Lock()
LOCAL_TZ = timezone(timedelta(hours=7), "Asia/Jakarta")


def write_log(event, **payload):
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now(LOCAL_TZ)
    record = {
        "timestamp": now_utc.isoformat(),
        "timestamp_local": now_local.isoformat(),
        "event": event,
        "model": MODEL_NAME,
        **payload,
    }

    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


write_log(
    "run_started",
    input_file=str(INPUT_FILE),
    output_file=str(OUTPUT_FILE),
    filtered_log=str(FILTERED_LOG),
    log_file=str(LOG_FILE),
    total_verses=len(verses),
    already_processed=len(processed),
    already_filtered=len(filtered_keys),
    remaining_to_process=len(pending_verses),
    parallel_workers=NUM_WORKERS,
)


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


def get_progress_snapshot(completed, total):
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


def print_progress_bar(completed, total):
    snapshot = get_progress_snapshot(completed, total)
    new_results = len(results) - initial_results_count
    bar = build_progress_bar(completed, total)
    message = (
        f"\r{bar} {completed}/{total} "
        f"| generated {new_results} "
        f"| ETA {format_duration(snapshot['eta_seconds'])} "
        f"| finish {snapshot['estimated_finish_local'].strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )
    print(message, end="", flush=True)


# =====================================
# PROMPT BUILDER
# =====================================
def build_messages(surah, ayat, text, theme_path):
    if theme_path:
        theme_string = " -> ".join(theme_path)
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

The verse belongs to one thematic category path.
This thematic path provides contextual orientation only.
It must NOT introduce information beyond the verse text.

The question must simulate a natural user query.
The user does NOT see the verse.

Strict Rules:
- Do NOT refer to "ayat ini" or similar phrases.
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
Theme path:
{theme_string}

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
        {"role": "user", "content": user_prompt},
    ]


def parse_json_content(content):
    cleaned = content.strip()

    if cleaned == "INSUFFICIENT_EVIDENCE":
        return cleaned

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# =====================================
# GENERATION FUNCTION
# =====================================
def generate_gold(entry):
    global total_tokens_used
    global insufficient_count
    global filtered_count
    global parse_error_count

    surah = entry["surah"]
    ayat = entry["ayat"]
    text = entry["text_id"]
    theme_paths = entry.get("theme_paths", [])
    theme_path = theme_paths[0] if theme_paths else []
    item_start_time = time.time()

    if (surah, ayat) in processed:
        write_log(
            "skipped_already_processed",
            surah=surah,
            ayat=ayat,
            duration_seconds=time.time() - item_start_time,
            elapsed_seconds=time.time() - start_time,
        )
        return None

    attempt = 0

    while True:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=build_messages(surah, ayat, text, theme_path),
                temperature=0.0,
                max_tokens=400,
            )
            break
        except Exception as e:
            if not is_retryable_error(e):
                message = str(e)

                if "content_filter" in message:
                    filtered_count += 1
                    filtered_log.append({"surah": surah, "ayat": ayat})
                    write_log(
                        "content_filtered",
                        surah=surah,
                        ayat=ayat,
                        error=message,
                        duration_seconds=time.time() - item_start_time,
                        elapsed_seconds=time.time() - start_time,
                    )
                    return None

                write_log(
                    "error",
                    surah=surah,
                    ayat=ayat,
                    error_type=type(e).__name__,
                    error=message,
                    duration_seconds=time.time() - item_start_time,
                    elapsed_seconds=time.time() - start_time,
                )
                print(f"Error on {surah}:{ayat} -> {e}")
                return None

            attempt += 1
            wait_seconds = retry_delay_seconds(attempt)

            write_log(
                "retry_waiting",
                surah=surah,
                ayat=ayat,
                attempt=attempt,
                wait_seconds=wait_seconds,
                elapsed_seconds=time.time() - start_time,
                error_type=type(e).__name__,
                error=str(e),
            )
            print(
                f"\nRetry jaringan/API {surah}:{ayat} "
                f"(attempt {attempt}, tunggu {format_duration(wait_seconds)})"
            )
            time.sleep(wait_seconds)

    try:
        if response.usage and response.usage.total_tokens:
            total_tokens_used += response.usage.total_tokens

        content = response.choices[0].message.content.strip()
        parsed = parse_json_content(content)

        if parsed == "INSUFFICIENT_EVIDENCE":
            insufficient_count += 1
            write_log(
                "insufficient_evidence",
                surah=surah,
                ayat=ayat,
                duration_seconds=time.time() - item_start_time,
                elapsed_seconds=time.time() - start_time,
            )
            return {
                "surah": surah,
                "ayat": ayat,
                "status": "INSUFFICIENT_EVIDENCE",
                "model": MODEL_NAME,
                "theme_path_used": theme_path,
            }

        parsed["surah"] = surah
        parsed["ayat"] = ayat
        parsed["model"] = MODEL_NAME
        parsed["theme_path_used"] = theme_path

        write_log(
            "generated",
            surah=surah,
            ayat=ayat,
            difficulty=parsed.get("difficulty"),
            total_tokens=response.usage.total_tokens if response.usage else None,
            duration_seconds=time.time() - item_start_time,
            elapsed_seconds=time.time() - start_time,
        )

        return parsed

    except Exception as e:
        message = str(e)

        if "content_filter" in message:
            filtered_count += 1
            filtered_log.append({"surah": surah, "ayat": ayat})
            write_log(
                "content_filtered",
                surah=surah,
                ayat=ayat,
                error=message,
                duration_seconds=time.time() - item_start_time,
                elapsed_seconds=time.time() - start_time,
            )
            return None

        if "JSON" in type(e).__name__ or isinstance(e, json.JSONDecodeError):
            parse_error_count += 1

        write_log(
            "error",
            surah=surah,
            ayat=ayat,
            error_type=type(e).__name__,
            error=message,
            duration_seconds=time.time() - item_start_time,
            elapsed_seconds=time.time() - start_time,
        )
        print(f"Error on {surah}:{ayat} -> {e}")
        return None


# =====================================
# PARALLEL EXECUTION
# =====================================
run_total = len(pending_verses)

if run_total == 0:
    print("No remaining verses to process.")
    write_log("nothing_to_process")
else:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(generate_gold, v): v for v in pending_verses}

        for completed_count, future in enumerate(as_completed(futures), 1):
            result = future.result()

            if result:
                results.append(result)

            save_json_atomic(OUTPUT_FILE, results)
            save_json_atomic(FILTERED_LOG, filtered_log)
            print_progress_bar(completed_count, run_total)

            if completed_count % 25 == 0 or completed_count == run_total:
                snapshot = get_progress_snapshot(completed_count, run_total)
                new_results = len(results) - initial_results_count

                print("\n========== PROGRESS ==========")
                print(f"Model: {MODEL_NAME}")
                print(f"Parallel workers: {NUM_WORKERS}")
                print(f"Completed: {completed_count}/{run_total}")
                print(f"Generated this run: {new_results}")
                print(f"Total results saved: {len(results)}")
                print(f"Elapsed: {format_duration(snapshot['elapsed'])}")
                print(f"Avg per request: {snapshot['avg_time']:.2f} sec")
                print(f"ETA: {format_duration(snapshot['eta_seconds'])}")
                print(
                    "Estimated finish: "
                    f"{snapshot['estimated_finish_local'].strftime('%Y-%m-%d %H:%M:%S %Z')}"
                )
                print(f"Tokens used: {total_tokens_used}")
                print(f"Filtered: {filtered_count}")
                print(f"Insufficient: {insufficient_count}")
                print(f"Parse errors: {parse_error_count}")
                print("================================")

                write_log(
                    "progress_saved",
                    completed=completed_count,
                    total_to_process=run_total,
                    generated_this_run=new_results,
                    total_results_saved=len(results),
                    elapsed_seconds=snapshot["elapsed"],
                    elapsed_minutes=snapshot["elapsed"] / 60,
                    average_seconds_per_request=snapshot["avg_time"],
                    eta_seconds=snapshot["eta_seconds"],
                    eta_minutes=snapshot["eta_seconds"] / 60,
                    estimated_finish_local=snapshot["estimated_finish_local"].isoformat(),
                    parallel_workers=NUM_WORKERS,
                    total_tokens_used=total_tokens_used,
                    filtered=filtered_count,
                    insufficient=insufficient_count,
                    parse_errors=parse_error_count,
                )

    print()

# =====================================
# FINAL SAVE & SUMMARY
# =====================================
save_json_atomic(OUTPUT_FILE, results)
save_json_atomic(FILTERED_LOG, filtered_log)

total_time = time.time() - start_time
average_time = total_time / run_total if run_total else 0
new_results = len(results) - initial_results_count

print("\n========== FINAL SUMMARY ==========")
print(f"Model: {MODEL_NAME}")
print(f"Parallel workers: {NUM_WORKERS}")
print(f"Total runtime: {format_duration(total_time)}")
print(f"Requests completed this run: {run_total}")
print(f"Generated this run: {new_results}")
print(f"Total results saved: {len(results)}")
print(f"Average time per request: {average_time:.2f} sec")
print(f"Total tokens used: {total_tokens_used}")
print(f"Filtered verses: {filtered_count}")
print(f"Insufficient evidence: {insufficient_count}")
print(f"Parse errors: {parse_error_count}")
print(f"Output file: {OUTPUT_FILE}")
print("===================================")

write_log(
    "run_finished",
    finished_at_local=datetime.now(LOCAL_TZ).isoformat(),
    total_runtime_seconds=total_time,
    total_runtime_minutes=total_time / 60,
    requests_completed_this_run=run_total,
    generated_this_run=new_results,
    total_results_saved=len(results),
    average_seconds_per_request=average_time,
    parallel_workers=NUM_WORKERS,
    total_tokens_used=total_tokens_used,
    filtered=filtered_count,
    insufficient=insufficient_count,
    parse_errors=parse_error_count,
    output_file=str(OUTPUT_FILE),
)
