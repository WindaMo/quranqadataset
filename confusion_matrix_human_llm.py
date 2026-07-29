import json
import pandas as pd

FILE = "validation_log_3models_human_MPTP.jsonl"

rows = []
with open(FILE,"r",encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

def encode(x):
    if x == "SUPPORTED":
        return 1
    if x == "NOT_SUPPORTED":
        return 0
    return None

for col in ["A","B","C","D","E"]:
    if col in df.columns:
        df[col] = df[col].apply(encode)

# keep rows with human labels
df = df.dropna(subset=["D","E"])

# LLM majority
df["llm_majority"] = (df[["A","B","C"]].sum(axis=1) >= 2).astype(int)

# Human majority
df["human_majority"] = (df[["D","E"]].sum(axis=1) >= 1).astype(int)

confusion = pd.crosstab(
    df["llm_majority"],
    df["human_majority"],
    rownames=["LLM"],
    colnames=["Human"]
)

print(confusion)