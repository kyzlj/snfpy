import pandas as pd
from sklearn.cluster import SpectralClustering
import networkx as nx
import matplotlib.pyplot as plt
from snf import datasets, compute, metrics
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# 加载数据
gene_data = pd.read_csv("snf/tests/data/GBM/GLIO_Gene_Expression.txt", sep="\t", index_col=0)
methy_data = pd.read_csv("snf/tests/data/GBM/GLIO_Methy_Expression.txt", sep="\t", index_col=0)
mirna_data = pd.read_csv("snf/tests/data/GBM/GLIO_Mirna_Expression.txt", sep="\t", index_col=0)
survival_data = pd.read_csv("snf/tests/data/GBM/GLIO_Survival.txt", sep="\t", index_col=0)

# 将原始数据数组转换为相似性矩阵（亲和力矩阵）
affinities1 = compute.make_affinity(gene_data, metric='euclidean', K=20, mu=0.5)
affinities2 = compute.make_affinity(methy_data, metric='euclidean', K=20, mu=0.5)
affinities3 = compute.make_affinity(mirna_data, metric='euclidean', K=20, mu=0.5)

# 融合相似性矩阵
fused = compute.snf(affinities1,affinities1,affinities1,K=20)

# 确定最佳聚类数（这里假设为3）
n_clusters = 3
clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
labels = clustering.fit_predict(fused)

# 创建网络图
G = nx.Graph()

# 添加节点并设置节点大小（根据生存数据调整节点大小）
for i in range(len(survival_data)):
    G.add_node(i, size=survival_data.iloc[i, 0], label=labels[i])

# 添加边并设置边权重（根据相似性矩阵调整边粗细）
for i in range(len(fused)):
    for j in range(i + 1, len(fused)):
        if fused[i, j] > 0.5:  # 设定一个相似性阈值
            G.add_edge(i, j, weight=fused[i, j])

# 可视化网络图
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 10))

# 绘制节点
node_sizes = [G.nodes[i]['size'] * 10 for i in G.nodes]  # 根据生存情况调整节点大小
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=labels, cmap=plt.cm.viridis)

# 绘制边
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]  # 根据相似性调整边粗细
nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights)

# 绘制标签
nx.draw_networkx_labels(G, pos)

plt.title("Patient Similarity Network")
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="Cluster Label")
plt.show()


