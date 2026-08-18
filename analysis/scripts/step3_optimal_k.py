

INPUT_CSV  = "combined_75.csv"
OUTPUT_DIR = "."

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



df = pd.read_csv(INPUT_CSV)
feature_cols = (
    [f"Q1_{L}" for L in "ABCDE"]
    + [f"Q2_{L}" for L in "ABCD"]
    + [f"Q3_{L}" for L in "ABC"]
    + [f"Q4_{L}" for L in "ABCDEF"]
)
X = df[feature_cols].values
K_RANGE = range(2, 11)   


print("=" * 55)
print("[A] Minimal prototype (single run per k)")
print("=" * 55)
print(f"{'k':>3} {'WCSS':>10} {'Silhouette':>12}")
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    print(f"{k:>3} {km.inertia_:>10.2f} {sil:>12.4f}")


N_SEEDS = 30   

records = []
for k in K_RANGE:
    for seed in range(N_SEEDS):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        records.append({
            "k": k, "seed": seed,
            "wcss": km.inertia_,
            "sil": silhouette_score(X, km.labels_),
        })

runs = pd.DataFrame(records)
summary = runs.groupby("k").agg(
    wcss_mean=("wcss", "mean"), wcss_std=("wcss", "std"),
    sil_mean=("sil", "mean"),   sil_std=("sil", "std"),
).round(4)

print("\n" + "=" * 55)
print(f"[B] Multi-seed averages (N={N_SEEDS} seeds per k)")
print("=" * 55)
print(summary.to_string())


best_k_sil = summary["sil_mean"].idxmax()
print(f"\n>>> Silhouette 峰值在 k = {best_k_sil} "
      f"(mean = {summary.loc[best_k_sil, 'sil_mean']:.4f})")

wcss = summary["wcss_mean"].values
second_diff = np.diff(wcss, n=2)
elbow_idx = np.argmax(second_diff) + 1   
best_k_elbow = list(K_RANGE)[elbow_idx]
print(f">>> Elbow 检测（二阶差分近似）在 k = {best_k_elbow}")



os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.errorbar(summary.index, summary["wcss_mean"], yerr=summary["wcss_std"],
            marker="o", capsize=3, color="#2b6cb0")
ax.axvline(6, ls="--", color="green", alpha=0.6, label="k=6 (theoretical)")
ax.axvline(best_k_elbow, ls=":", color="red", alpha=0.6,
           label=f"k={best_k_elbow} (elbow detected)")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("WCSS (inertia)")
ax.set_title("Elbow Method")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.errorbar(summary.index, summary["sil_mean"], yerr=summary["sil_std"],
            marker="o", capsize=3, color="#805ad5")
ax.axvline(6, ls="--", color="green", alpha=0.6, label="k=6 (theoretical)")
ax.axvline(best_k_sil, ls=":", color="red", alpha=0.6,
           label=f"k={best_k_sil} (silhouette peak)")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Mean silhouette score")
ax.set_title("Silhouette Analysis")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle(f"K-Means diagnostics (n={len(df)}, 18 binary features, "
             f"{N_SEEDS} seeds/k)", y=1.02)
plt.tight_layout()

png_path = os.path.join(OUTPUT_DIR, "kmeans_diagnostics.png")
pdf_path = os.path.join(OUTPUT_DIR, "kmeans_diagnostics.pdf")
csv_path = os.path.join(OUTPUT_DIR, "kmeans_diagnostics_table.csv")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()
summary.to_csv(csv_path)

print(f"\n已保存 → {png_path}")
print(f"        → {pdf_path}")
print(f"        → {csv_path}")


print("\n" + "=" * 55)
print("[D] Verdict")
print("=" * 55)
if best_k_sil == 6 and best_k_elbow == 6:
    verdict = "两个指标都指向 k=6 → 强证据支持 6-profile 理论。"
elif best_k_sil == 6 or best_k_elbow == 6:
    which = "silhouette" if best_k_sil == 6 else "elbow"
    other_k = best_k_elbow if best_k_sil == 6 else best_k_sil
    verdict = (f"仅 {which} 指向 k=6，另一个指向 k={other_k}。"
               "部分支持理论，讨论章需要展开。")
else:
    verdict = (f"两个指标都未指向 k=6（elbow → {best_k_elbow}, "
               f"silhouette → {best_k_sil}）。这是需要在讨论章展开的实证发现。")
print(verdict)
