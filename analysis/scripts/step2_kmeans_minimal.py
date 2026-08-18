
INPUT_CSV  = "combined_75.csv"  
OUTPUT_DIR = "."                 

import os
import pandas as pd
from sklearn.cluster import KMeans




df = pd.read_csv(INPUT_CSV)

feature_cols = (
    [f"Q1_{L}" for L in "ABCDE"]
    + [f"Q2_{L}" for L in "ABCD"]
    + [f"Q3_{L}" for L in "ABC"]
    + [f"Q4_{L}" for L in "ABCDEF"]
)
X = df[feature_cols].values
print(f"特征矩阵形状: {X.shape}   (75 × 18)")


kmeans = KMeans(
    n_clusters=6,     
    n_init=10,        
    random_state=42,  
)
df["cluster"] = kmeans.fit_predict(X)



print(f"\n=== 簇大小分布 ===")
print(df["cluster"].value_counts().sort_index().to_string())

print(f"\n=== 每簇的答题倾向（列的均值 = 该选项被选中的比例）===")
cluster_profile = df.groupby("cluster")[feature_cols].mean().round(2)
print(cluster_profile.to_string())

print(f"\n=== 每簇最突出的 3 个特征（比例 ≥ 0.6）===")
for c in sorted(df["cluster"].unique()):
    row = cluster_profile.loc[c]
    dominant = row[row >= 0.6].sort_values(ascending=False)
    n = (df["cluster"] == c).sum()
    print(f"  Cluster {c} (n={n}): {dict(dominant)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "combined_75_clustered.csv")
df.to_csv(out_path, index=False)
print(f"\n已保存 → {out_path}")
