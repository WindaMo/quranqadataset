import json
import pandas as pd

validation_file = "validation_log_3models_MPTP.jsonl"
human_file = "evaluation_values.csv"

# read CSV
df = pd.read_csv(human_file, sep=";")

# clean column names
df.columns = df.columns.str.strip()

# pivot evaluator rows into columns
pivot = df.pivot_table(
    index="ID",
    columns="nama_evaluator",
    values="verdict",
    aggfunc="first"
).reset_index()

# split surah and ayat
pivot["surah"] = pivot["ID"].str.split(":").str[0].astype(int)
pivot["ayat"] = pivot["ID"].str.split(":").str[1].astype(int)

# create dictionary for fast lookup
human_dict = {
    (row.surah, row.ayat): {
        "D": row.get("D"),
        "E": row.get("E")
    }
    for _, row in pivot.iterrows()
}

output = []

# merge with LLM validation log
with open(validation_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        key = (obj["surah"], obj["ayat"])

        if key in human_dict:
            obj["D"] = human_dict[key]["D"]
            obj["E"] = human_dict[key]["E"]

        output.append(obj)

# export JSONL
with open("validation_log_3models_human_MPTP.jsonl", "w", encoding="utf-8") as f:
    for row in output:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Export complete: validation_log_3models_human_MPTP.jsonl")