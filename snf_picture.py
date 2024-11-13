import pandas as pd
from sklearn.cluster import SpectralClustering
import networkx as nx
import matplotlib.pyplot as plt
from snf import datasets, compute, metrics
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载数据
gene_data = pd.read_csv("snf/tests/data/GBM/GLIO_Gene_Expression.csv", sep=",", index_col=0)
methy_data = pd.read_csv("snf/tests/data/GBM/GLIO_Methy_Expression.csv", sep=",", index_col=0)
mirna_data = pd.read_csv("snf/tests/data/GBM/GLIO_Mirna_Expression.csv", sep=",", index_col=0)
survival_data = pd.read_csv("snf/tests/data/GBM/GLIO_Survival.csv", sep=",", index_col=0)

# 输出数据集的基本信息
print("Gene Expression Data Shape:", gene_data.shape)
print("Methylation Data Shape:", methy_data.shape)
print("miRNA Data Shape:", mirna_data.shape)
print("Survival Data Shape:", survival_data.shape)

# 转换为相似性矩阵
affinities1 = compute.make_affinity(gene_data, metric='euclidean', K=20, mu=0.5)
affinities2 = compute.make_affinity(methy_data, metric='euclidean', K=20, mu=0.5)
affinities3 = compute.make_affinity(mirna_data, metric='euclidean', K=20, mu=0.5)

# 融合相似性矩阵
fused = compute.snf(affinities1, affinities2, affinities3, K=20)
print("\nFused Affinity Matrix Shape:", fused.shape)

# 聚类
n_clusters = 3
clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
labels = clustering.fit_predict(fused)
print("\nCluster Labels:", labels)

# 创建网络图
G = nx.Graph()

# 添加节点和标签
for i in range(len(survival_data)):
    G.add_node(i, size=survival_data.iloc[i, 0], label=labels[i])

print("\nNetwork Nodes:", G.nodes(data=True))

# 添加边
for i in range(len(fused)):
    for j in range(i + 1, len(fused)):
        if fused[i, j] > 0.5:  # 相似性阈值
            G.add_edge(i, j, weight=fused[i, j])

print("\nNetwork Edges:", G.edges(data=True))

# 可视化网络图
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 10))

# 绘制节点
node_sizes = [G.nodes[i]['size'] * 10 for i in G.nodes]
node_colors = [G.nodes[i]['label'] for i in G.nodes]  # 使用标签作为颜色

# 创建 mappable 对象并绘制节点
sc = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis)

# 绘制边
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges] if edges else []  # 确保有边的权重
nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color="gray")  # 设置边颜色为灰色

# 绘制标签
nx.draw_networkx_labels(G, pos)

# 添加颜色条
plt.colorbar(sc, label="Cluster Label")  # 使用节点的颜色映射

plt.title("Patient Similarity Network")
plt.show()
