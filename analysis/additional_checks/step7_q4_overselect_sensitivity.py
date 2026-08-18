# step7_q4_overselect_sensitivity.py
# [A] Minimal check: effect of excluding the 3 English-arm
#     over-selectors (>3 options on Q4) on Q4 frequencies.

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "combined_75.csv"
df = pd.read_csv(DATA_PATH)
print(df.columns.tolist())   

q4_cols = [c for c in df.columns if c.startswith("Q4")]  
arm_col = "source"        

n_selected = df[q4_cols].sum(axis=1)
mask = n_selected > 3                     
print("over-selectors:", mask.sum(), "| arm:", df.loc[mask, arm_col].tolist())
print("their pick counts:", n_selected[mask].tolist())

full    = df[q4_cols].mean() * 100
trimmed = df[~mask][q4_cols].mean() * 100

out = pd.DataFrame({
    "full_pct":    full.round(1),
    "trimmed_pct": trimmed.round(1),
    "shift":       (full - trimmed).abs().round(1),
}).sort_values("full_pct", ascending=False)

print(out)
print("\nmax shift:", out["shift"].max())
print("ranking unchanged:",
      list(out.index) == list(out.sort_values("trimmed_pct", ascending=False).index))
