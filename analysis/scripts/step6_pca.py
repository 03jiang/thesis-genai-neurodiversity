

INPUT_CSV  = "combined_75_dual_labels.csv"
OUTPUT_DIR = "."


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes




df = pd.read_csv(INPUT_CSV)
feature_cols = (
    [f"Q1_{L}" for L in "ABCDE"]
    + [f"Q2_{L}" for L in "ABCD"]
    + [f"Q3_{L}" for L in "ABC"]
    + [f"Q4_{L}" for L in "ABCDEF"]
)
X = df[feature_cols].values



pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
var_pct = pca.explained_variance_ratio_ * 100

print("=" * 55)
print("[A] PCA minimal prototype")
print("=" * 55)
print(f"投影后 shape: {X_2d.shape}")
print(f"PC1 保留方差: {var_pct[0]:.1f}%")
print(f"PC2 保留方差: {var_pct[1]:.1f}%")
print(f"合计: {var_pct.sum():.1f}%   "
      f"(其余 {100 - var_pct.sum():.1f}% 落在 PC3..PC18)")


CLUSTER_COLORS = plt.cm.tab10(np.linspace(0, 1, 6))
Q5_MARKERS = {"Yes": "^", "No": "o", "Prefer": "s"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

def plot_one(ax, method_col, method_name, centroids_2d=None):
    for c in sorted(df[method_col].unique()):
        mask_c = df[method_col] == c
        for q5_val, marker in Q5_MARKERS.items():
            mask = mask_c & (df["Q5_label"] == q5_val)
            if not mask.any():
                continue
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[CLUSTER_COLORS[int(c)]], marker=marker,
                       s=80, alpha=0.75, edgecolors="white", linewidth=0.8)

    if centroids_2d is not None:
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                   marker="X", s=200, c="black",
                   edgecolors="yellow", linewidth=1.5, zorder=10)

    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}% variance)")
    ax.set_title(f"{method_name} clusters (k=6)")
    ax.grid(alpha=0.25)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)

km = KMeans(n_clusters=6, n_init=10, random_state=42).fit(X)
km_centroids_2d = pca.transform(km.cluster_centers_)
plot_one(axes[0], "kmeans_cluster", "K-Means", km_centroids_2d)

kmo = KModes(n_clusters=6, init="Huang", n_init=10,
             random_state=42, verbose=0).fit(X)
kmo_centroids_2d = pca.transform(kmo.cluster_centroids_)
plot_one(axes[1], "kmodes_cluster", "K-Modes", kmo_centroids_2d)

cluster_legend = [Patch(facecolor=CLUSTER_COLORS[i], label=f"Cluster {i}")
                  for i in range(6)]
q5_legend = [Line2D([0], [0], marker=m, color="w",
                     markerfacecolor="gray", markersize=10, label=f"Q5: {q}")
             for q, m in Q5_MARKERS.items()]
centroid_legend = [Line2D([0], [0], marker="X", color="w",
                           markerfacecolor="black", markeredgecolor="yellow",
                           markersize=13, label="Projected centroid/mode")]

fig.legend(handles=cluster_legend + q5_legend + centroid_legend,
           loc="center right", bbox_to_anchor=(1.13, 0.5),
           frameon=True, fontsize=9)

plt.suptitle(f"PCA projection of clustered survey data (n={len(df)}, "
             f"first 2 components = {var_pct.sum():.1f}% variance)", y=1.02)
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
clusters_png = os.path.join(OUTPUT_DIR, "pca_clusters.png")
clusters_pdf = os.path.join(OUTPUT_DIR, "pca_clusters.pdf")
plt.savefig(clusters_png, dpi=150, bbox_inches="tight")
plt.savefig(clusters_pdf, bbox_inches="tight")
plt.close()



loadings = pd.DataFrame(pca.components_.T,
                         index=feature_cols,
                         columns=["PC1", "PC2"])

fig, ax = plt.subplots(figsize=(6, 8))
im = ax.imshow(loadings.values, cmap="RdBu_r", aspect="auto",
               vmin=-0.6, vmax=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["PC1", "PC2"])
ax.set_yticks(range(len(feature_cols)))
ax.set_yticklabels(feature_cols, fontsize=9)

for i in range(len(feature_cols)):
    for j in range(2):
        val = loadings.iloc[i, j]
        color = "white" if abs(val) > 0.35 else "black"
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                color=color, fontsize=8)

for boundary in [4.5, 8.5, 11.5]:      # Q1|Q2|Q3|Q4 之间
    ax.axhline(boundary, color="black", lw=0.8)

plt.colorbar(im, ax=ax, label="Loading")
ax.set_title("Loadings of the 18 binary variables on PC1 and PC2", fontsize=11)
plt.tight_layout()

loadings_png = os.path.join(OUTPUT_DIR, "pca_loadings.png")
loadings_pdf = os.path.join(OUTPUT_DIR, "pca_loadings.pdf")
plt.savefig(loadings_png, dpi=150, bbox_inches="tight")
plt.savefig(loadings_pdf, bbox_inches="tight")
plt.close()


print("\n" + "=" * 55)
print("[E] PC1 与 PC2 的语义（top ±3 特征）")
print("=" * 55)

for pc in ["PC1", "PC2"]:
    top_pos = loadings[pc].nlargest(3)
    top_neg = loadings[pc].nsmallest(3)
    vp = var_pct[0] if pc == "PC1" else var_pct[1]
    print(f"\n{pc}（保留 {vp:.1f}% 方差）:")
    print(f"  正方向: {dict(top_pos.round(3))}")
    print(f"  负方向: {dict(top_neg.round(3))}")

print(f"\n已保存:")
print(f"  {clusters_png} / {clusters_pdf}   —— 主可视化")
print(f"  {loadings_png} / {loadings_pdf}   —— 解读辅助")
print(pca.explained_variance_ratio_[:2])
print(pca.explained_variance_ratio_[:2].sum())