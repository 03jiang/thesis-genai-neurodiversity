

INPUT_CSV  = "combined_75_dual_labels.csv"   # Step 4 生成的双标签数据
OUTPUT_DIR = "."

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency



df = pd.read_csv(INPUT_CSV)
df_v = df.dropna(subset=["Q5_label"]).copy()
print(f"有效样本 (Q5 非空): n = {len(df_v)}")
print(f"Q5 分布: {dict(df_v['Q5_label'].value_counts())}")


def cramers_v(ct):
    """Cramér's V —— chi-square 派生效应量。"""
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    n = ct.values.sum() if hasattr(ct, "values") else ct.sum()
    r, c = ct.shape
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))


print("\n" + "=" * 55)
print("[A] Minimal prototype: K-Means clusters × Q5")
print("=" * 55)
ct_km = pd.crosstab(df_v["kmeans_cluster"], df_v["Q5_label"])
print("列联表:")
print(ct_km.to_string())
chi2, p, dof, expected = chi2_contingency(ct_km)
v = cramers_v(ct_km)
print(f"\nChi-square = {chi2:.3f}, dof = {dof}, p = {p:.4f}")
print(f"Cramér's V = {v:.3f}")


def small_sample_diagnostics(ct):
    _, _, _, expected = chi2_contingency(ct)
    pct_below_5 = (expected < 5).sum() / expected.size * 100
    return pct_below_5, expected.min()

def monte_carlo_pvalue(ct, n_simulations=10000, random_state=42):
    rng = np.random.default_rng(random_state)
    observed_chi2, _, _, _ = chi2_contingency(ct.values, correction=False)
    row_sums = ct.sum(axis=1).values
    col_sums = ct.sum(axis=0).values
    n = row_sums.sum()

    p_null = np.outer(row_sums, col_sums) / (n ** 2)
    p_null_flat = p_null.flatten()

    count_extreme = 0
    for _ in range(n_simulations):
        sample_flat = rng.multinomial(n, p_null_flat)
        sample = sample_flat.reshape(ct.shape)
        try:
            sim_chi2, _, _, _ = chi2_contingency(sample, correction=False)
            if sim_chi2 >= observed_chi2:
                count_extreme += 1
        except ValueError:
            continue
    return (count_extreme + 1) / (n_simulations + 1)   # Laplace 修正避免 p=0


pct5, min_exp = small_sample_diagnostics(ct_km)
print(f"\n小样本诊断:")
print(f"  期望频次 < 5 的格子占比: {pct5:.1f}%")
print(f"  最小期望频次: {min_exp:.2f}")
if pct5 > 20:
    print(f"  ⚠️  >20% 格子 < 5，chi-square 渐近 p 值不可信")
    print(f"  → 计算 Monte Carlo p 值（10000 次模拟）...")
    p_mc = monte_carlo_pvalue(ct_km)
    print(f"  Monte Carlo p 值 = {p_mc:.4f}")
else:
    p_mc = p
    print(f"  期望频次充足，chi-square p 值可信")

print("\n" + "=" * 55)
print("[C] K-Modes clusters × Q5")
print("=" * 55)
ct_kmo = pd.crosstab(df_v["kmodes_cluster"], df_v["Q5_label"])
print("列联表:")
print(ct_kmo.to_string())

chi2_kmo, p_kmo, dof_kmo, _ = chi2_contingency(ct_kmo)
v_kmo = cramers_v(ct_kmo)
print(f"\nChi-square = {chi2_kmo:.3f}, dof = {dof_kmo}, p = {p_kmo:.4f}")
print(f"Cramér's V = {v_kmo:.3f}")

pct5_kmo, min_exp_kmo = small_sample_diagnostics(ct_kmo)
print(f"\n期望频次 < 5 的格子占比: {pct5_kmo:.1f}%")
if pct5_kmo > 20:
    p_mc_kmo = monte_carlo_pvalue(ct_kmo)
    print(f"Monte Carlo p 值 = {p_mc_kmo:.4f}")
else:
    p_mc_kmo = p_kmo


comparison = pd.DataFrame({
    "Method": ["K-Means", "K-Modes"],
    "Chi-square": [round(chi2, 3), round(chi2_kmo, 3)],
    "df": [dof, dof_kmo],
    "Monte Carlo p": [round(p_mc, 4), round(p_mc_kmo, 4)],
    "Cramér's V": [round(v, 3), round(v_kmo, 3)],
})

print("\n" + "=" * 55)
print("[D] 两种方法并列对比")
print("=" * 55)
print(comparison.to_string(index=False))

os.makedirs(OUTPUT_DIR, exist_ok=True)
comparison.to_csv(os.path.join(OUTPUT_DIR, "q5_comparison_table.csv"),
                  index=False)

q5_colors = {"Yes": "#e53e3e", "No": "#4a5568", "Prefer": "#a0aec0"}
q5_order = ["Yes", "No", "Prefer"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (name, ct, v_val, p_val) in zip(
        axes,
        [("K-Means", ct_km, v, p_mc), ("K-Modes", ct_kmo, v_kmo, p_mc_kmo)]):
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    cols = [c for c in q5_order if c in pct.columns]
    pct = pct[cols]
    pct.plot(kind="bar", stacked=True, ax=ax,
             color=[q5_colors[c] for c in cols],
             edgecolor="white", width=0.8)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Percentage within cluster (%)")
    ax.set_title(f"{name}: Q5 composition per cluster\n"
                 f"Cramér's V = {v_val:.3f}, MC p = {p_val:.4f}")
    ax.legend(title="Q5", loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    yes_baseline = (df_v["Q5_label"] == "Yes").mean() * 100
    ax.axhline(yes_baseline, ls="--", color="black", alpha=0.4,
               label=f"Overall Yes% = {yes_baseline:.1f}")

plt.suptitle(
    "Exploratory comparison: cluster membership vs neurodivergent self-identification",
    y=1.02
)
plt.tight_layout()

png_path = os.path.join(OUTPUT_DIR, "q5_comparison.png")
pdf_path = os.path.join(OUTPUT_DIR, "q5_comparison.pdf")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print(f"\n已保存 → {png_path}")
print(f"        → {pdf_path}")

print("\n" + "=" * 55)
print("VERDICT")
print("=" * 55)
alpha = 0.05
km_sig = p_mc < alpha
kmo_sig = p_mc_kmo < alpha
if km_sig and kmo_sig:
    print("Both clustering solutions show a statistically significant association with Q5.")
elif km_sig or kmo_sig:
    winner = "K-Means" if km_sig else "K-Modes"
    print(f"Only {winner} shows a statistically significant association with Q5.")
else:
    print("Neither clustering solution shows a statistically significant association with Q5.")
    print("The effect-size estimates are descriptive and should be interpreted cautiously.")
print(f"\nDescriptive Cramér's V: K-Means = {v:.3f}, "
      f"K-Modes = {v_kmo:.3f}")
