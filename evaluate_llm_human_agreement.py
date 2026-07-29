import json
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

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

# --------------------------------------------------
# PART 1: LLM METRICS (FULL DATASET)
# --------------------------------------------------

df_llm = df.dropna(subset=["A","B","C"]).copy()

df_llm["llm_sum"] = df_llm[["A","B","C"]].sum(axis=1)

df_llm["llm_majority"] = (df_llm["llm_sum"] >= 2).astype(int)
df_llm["llm_strict"] = (df_llm["llm_sum"] == 3).astype(int)

majority_rate = df_llm["llm_majority"].mean()
strict_rate = df_llm["llm_strict"].mean()

# --------------------------------------------------
# PART 2: HUMAN SUBSET
# --------------------------------------------------

df_human = df.dropna(subset=["D","E"]).copy()

N_human = len(df_human)

df_human["llm_sum"] = df_human[["A","B","C"]].sum(axis=1)
df_human["llm_majority"] = (df_human["llm_sum"] >= 2).astype(int)

df_human["human_majority"] = (df_human[["D","E"]].sum(axis=1) >= 1).astype(int)

# Human reliability
human_kappa = cohen_kappa_score(df_human["D"], df_human["E"])

# LLM vs Human
llm_human_agreement = (
    df_human["llm_majority"] == df_human["human_majority"]
).mean()

llm_human_kappa = cohen_kappa_score(
    df_human["llm_majority"],
    df_human["human_majority"]
)

# --------------------------------------------------
# PART 3: FLEISS κ (5 raters)
# --------------------------------------------------

df_fleiss = df.dropna(subset=["A","B","C","D","E"])

ratings = df_fleiss[["A","B","C","D","E"]].values

n_items = ratings.shape[0]
n_raters = 5

count_matrix = np.zeros((n_items,2))

for i in range(n_items):
    count_matrix[i,0] = np.sum(ratings[i]==0)
    count_matrix[i,1] = np.sum(ratings[i]==1)

p_j = np.sum(count_matrix,axis=0)/(n_items*n_raters)

P_i = (np.sum(count_matrix**2,axis=1)-n_raters)/(n_raters*(n_raters-1))

P_bar = np.mean(P_i)
P_e = np.sum(p_j**2)

fleiss_kappa = (P_bar-P_e)/(1-P_e)

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n===== LLM VALIDATION (FULL DATASET) =====")
print("Majority acceptance:", round(majority_rate,3))
print("Strict acceptance:", round(strict_rate,3))

print("\n===== HUMAN SUBSET =====")
print("Items:", N_human)
print("Human Cohen κ:", round(human_kappa,3))
print("LLM–Human agreement:", round(llm_human_agreement,3))
print("LLM–Human κ:", round(llm_human_kappa,3))

print("\n===== FLEISS κ (5 annotators) =====")
print("Items:", len(df_fleiss))
print("Fleiss κ:", round(fleiss_kappa,3))