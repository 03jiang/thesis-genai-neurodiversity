

INPUT_CSV  = "combined_75.csv"
OUTPUT_DIR = "."


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score



df = pd.read_csv(INPUT_CSV)
feature_cols = (
    [f"Q1_{L}" for L in "ABCDE"]
    + [f"Q2_{L}" for L in "ABCD"]
    + [f"Q3_{L}" for L in "ABC"]
    + [f"Q4_{L}" for L in "ABCDEF"]
)
X = df[feature_cols].values
print(f"数据: n={len(df)}, d={X.shape[1]}")


print("\n" + "=" * 55)
print("[A] Minimal prototype: K-Modes at k=6")
print("=" * 55)
km_modes = KModes(n_clusters=6, init="Huang", n_init=10,
                  random_state=42, verbose=0)
kmodes_labels = km_modes.fit_predict(X)
print(f"K-Modes cost (total Hamming mismatches): {km_modes.cost_:.0f}")
print(f"簇大小分布: {dict(pd.Series(kmodes_labels).value_counts().sort_index())}")


K_RANGE = range(2, 11)
N_SEEDS = 10

records = []
for k in K_RANGE:
    for seed in range(N_SEEDS):
        km = KModes(n_clusters=k, init="Huang", n_init=5,
                    random_state=seed, verbose=0).fit(X)
        labels = km.labels_
        try:
            sil = silhouette_score(X, labels, metric="hamming")
        except ValueError:
            sil = np.nan
        records.append({"k": k, "seed": seed, "cost": km.cost_, "sil": sil})

runs = pd.DataFrame(records)
summary = runs.groupby("k").agg(
    cost_mean=("cost", "mean"), cost_std=("cost", "std"),
    sil_mean=("sil", "mean"),   sil_std=("sil", "std"),
).round(4)

print("\n" + "=" * 55)
print(f"[B] Multi-k diagnostic (N={N_SEEDS} seeds/k)")
print("=" * 55)
print(summary.to_string())

best_k_sil = summary["sil_mean"].idxmax()
print(f"\n>>> Hamming silhouette 峰值在 k = {best_k_sil} "
      f"(mean = {summary.loc[best_k_sil, 'sil_mean']:.4f})")

cost = summary["cost_mean"].values
second_diff = np.diff(cost, n=2)
elbow_idx = np.argmax(second_diff) + 1
best_k_elbow = list(K_RANGE)[elbow_idx]
print(f">>> Cost 曲线肘部在 k = {best_k_elbow}")


print("\n" + "=" * 55)
print("[C] k=6 簇的典型画像（cluster mode vectors）")
print("=" * 55)

centroids = pd.DataFrame(km_modes.cluster_centroids_, columns=feature_cols)

for c in range(6):
    n = (kmodes_labels == c).sum()
    row = centroids.loc[c]
    active = row[row == 1].index.tolist()
    by_q = {}
    for col in active:
        q, opt = col.split("_")
        by_q.setdefault(q, []).append(opt)
    parts = [f"{q}={'/'.join(sorted(v))}" for q, v in sorted(by_q.items())]
    print(f"  Cluster {c} (n={n}): {' · '.join(parts)}")



kmeans = KMeans(n_clusters=6, n_init=10, random_state=42).fit(X)
ari = adjusted_rand_score(kmeans.labels_, kmodes_labels)

print("\n" + "=" * 55)
print("[D] K-Means vs K-Modes 一致性")
print("=" * 55)
print(f"Adjusted Rand Index (k=6): {ari:.4f}")
if ari > 0.7:
    print("→ 高一致性：两方法找到基本相同的结构")
elif ari > 0.4:
    print("→ 中等一致性：核心结构一致，边界受方法影响")
elif ari > 0.2:
    print("→ 弱一致性：结构对方法敏感")
else:
    print("→ 极低一致性：两方法看到完全不同的结构")

xtab = pd.crosstab(pd.Series(kmeans.labels_, name="K-Means"),
                    pd.Series(kmodes_labels, name="K-Modes"))
print(f"\n交叉表（行=K-Means, 列=K-Modes）:")
print(xtab.to_string())


os.makedirs(OUTPUT_DIR, exist_ok=True)

df["kmeans_cluster"] = kmeans.labels_
df["kmodes_cluster"] = kmodes_labels

dual_labels_path = os.path.join(OUTPUT_DIR, "combined_75_dual_labels.csv")
diag_csv_path = os.path.join(OUTPUT_DIR, "kmodes_diagnostics_table.csv")
df.to_csv(dual_labels_path, index=False)
summary.to_csv(diag_csv_path)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.errorbar(summary.index, summary["cost_mean"], yerr=summary["cost_std"],
            marker="o", capsize=3, color="#c05621")
ax.axvline(6, ls="--", color="green", alpha=0.6, label="k=6 (theoretical)")
ax.axvline(best_k_elbow, ls=":", color="red", alpha=0.6,
           label=f"k={best_k_elbow} (elbow detected)")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("K-Modes cost (Hamming mismatches)")
ax.set_title("K-Modes Cost Curve")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
ax.errorbar(summary.index, summary["sil_mean"], yerr=summary["sil_std"],
            marker="o", capsize=3, color="#38a169")
ax.axvline(6, ls="--", color="green", alpha=0.6, label="k=6 (theoretical)")
ax.axvline(best_k_sil, ls=":", color="red", alpha=0.6,
           label=f"k={best_k_sil} (silhouette peak)")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Mean silhouette score (Hamming)")
ax.set_title("K-Modes Silhouette Analysis")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle(f"K-Modes diagnostics (n={len(df)}, 18 binary features, "
             f"{N_SEEDS} seeds/k)", y=1.02)
plt.tight_layout()

png_path = os.path.join(OUTPUT_DIR, "kmodes_diagnostics.png")
pdf_path = os.path.join(OUTPUT_DIR, "kmodes_diagnostics.pdf")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print(f"\n已保存:")
print(f"  {dual_labels_path}   (Step 5/6 会用)")
print(f"  {png_path}")
print(f"  {pdf_path}")
print(f"  {diag_csv_path}")
