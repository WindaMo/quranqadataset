import json
import pandas as pd
import numpy as np
import krippendorff

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
    return np.nan

for col in ["A","B","C","D","E"]:
    if col in df.columns:
        df[col] = df[col].apply(encode)

# ------------------------------------------------
# FILTER ONLY ITEMS WITH HUMAN LABELS
# ------------------------------------------------

df5 = df.dropna(subset=["A","B","C","D","E"])

print("Items with full annotations:", len(df5))

# matrix must be annotators × items
data = df5[["A","B","C","D","E"]].to_numpy().T

alpha_all = krippendorff.alpha(
    reliability_data=data,
    level_of_measurement="nominal"
)

print("\nKrippendorff α (A,B,C,D,E) =", round(alpha_all,3))