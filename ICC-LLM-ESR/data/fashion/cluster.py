import pickle
import numpy as np
import umap
from sklearn.cluster import DBSCAN

# 1. 加载嵌入数据（与你的代码一致）
with open('./handled/pca64_itm_emb_np.pkl', 'rb') as f:
    embeddings = pickle.load(f)  # 形状：(num_items, 64)
print(f"嵌入数据加载完成，形状: {embeddings.shape}")

# 2. 最佳参数（从搜索结果中提取）
best_params = {
    'umap_n_neighbors': 8,
    'umap_min_dist': 0.025,
    'dbscan_eps': 0.13,
    'dbscan_min_samples': 8
}

# 3. UMAP降维到8维（用于后续计算聚类中心）
reducer = umap.UMAP(
    n_components=8,
    n_neighbors=best_params['umap_n_neighbors'],
    min_dist=best_params['umap_min_dist'],
    metric='cosine',
    random_state=42
)
umap_8d_embeds = reducer.fit_transform(embeddings)  # 形状：(num_items, 8)

# 4. DBSCAN聚类，得到每个item的标签（-1为噪声）
dbscan = DBSCAN(
    eps=best_params['dbscan_eps'],
    min_samples=best_params['dbscan_min_samples'],
    metric='euclidean'
)
item_cluster_labels = dbscan.fit_predict(umap_8d_embeds)  # 形状：(num_items,)，每个元素为聚类ID或-1

# 5. 计算每个非噪声聚类的8维中心（聚类中心 = 该类所有item的UMAP 8维嵌入的均值）
unique_labels = np.unique(item_cluster_labels)
cluster_centers_8d = {}  # 键：聚类ID（非-1），值：8维中心数组

for label in unique_labels:
    if label == -1:  # 跳过噪声
        continue
    # 提取该聚类下所有item的8维UMAP嵌入
    cluster_embeds = umap_8d_embeds[item_cluster_labels == label]
    # 计算均值作为中心
    cluster_center = np.mean(cluster_embeds, axis=0)  # 形状：(8,)
    cluster_centers_8d[label] = cluster_center

# 6. 保存结果为pkl文件（输出到data/fashion/handled/目录）
output_dir = './handled/'

# 保存每个item的聚类标签（含噪声标记）
with open(f'{output_dir}/item_cluster_labels.pkl', 'wb') as f:
    pickle.dump(item_cluster_labels, f)

# 保存8维聚类中心（仅非噪声聚类）
with open(f'{output_dir}/cluster_centers_8d.pkl', 'wb') as f:
    pickle.dump(cluster_centers_8d, f)

print("聚类结果文件生成完成：")
print(f"- item_cluster_labels.pkl 形状: {item_cluster_labels.shape}")
print(f"- cluster_centers_8d.pkl 包含 {len(cluster_centers_8d)} 个非噪声聚类中心")