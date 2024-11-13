import pandas as pd
from sklearn.cluster import SpectralClustering
import networkx as nx
import matplotlib.pyplot as plt
from snf import datasets, compute, metrics
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载数据
gene_data = pd.read_csv("snf/tests/data/GBM/GLIO_Gene_Expression.csv", sep=",", index_col=0).transpose()
methy_data = pd.read_csv("snf/tests/data/GBM/GLIO_Methy_Expression.csv", sep=",", index_col=0).transpose()
mirna_data = pd.read_csv("snf/tests/data/GBM/GLIO_Mirna_Expression.csv", sep=",", index_col=0).transpose()
survival_data = pd.read_csv("snf/tests/data/GBM/GLIO_Survival.csv", sep=",", index_col=0)

# 输出数据集的基本信息和前几行数据
print("Gene Expression Data:")
print(gene_data.shape)
# print(gene_data.head())

print("\nMethylation Data:")
print(methy_data.shape)
# print(methy_data.head())

print("\nmiRNA Data:")
print(mirna_data.shape)
# print(mirna_data.head())

print("\nSurvival Data:")
print(survival_data.shape)
# print(survival_data.head())

# 将原始数据数组转换为相似性矩阵（亲和力矩阵）
affinities1 = compute.make_affinity(gene_data, metric='euclidean', K=20, mu=0.5)
affinities2 = compute.make_affinity(methy_data, metric='euclidean', K=20, mu=0.5)
affinities3 = compute.make_affinity(mirna_data, metric='euclidean', K=20, mu=0.5)

# 输出相似性矩阵的基本信息
print("\nAffinity Matrix 1 (Gene Expression):")
print(affinities1.shape)

print("\nAffinity Matrix 2 (Methylation):")
print(affinities2.shape)

print("\nAffinity Matrix 3 (miRNA):")
print(affinities3.shape)

# 融合相似性矩阵
fused = compute.snf(affinities1, affinities2, affinities3, K=20)

# 输出融合后的相似性矩阵的基本信息
print("\nFused Affinity Matrix:")
print(fused.shape)

# 确定最佳聚类数（这里假设为3）
n_clusters = 3
clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
labels = clustering.fit_predict(fused)

# 输出聚类标签
print("\nCluster Labels:")
print(labels)

# 创建网络图
G = nx.Graph()

# 添加节点并设置节点大小（根据生存数据调整节点大小）
for i in range(len(survival_data)):
    G.add_node(i, size=survival_data.iloc[i, 0], label=labels[i])

# 输出网络中节点的基本信息
print("\nNetwork Nodes:")
print(G.nodes(data=True))

# 添加边并设置边权重（根据相似性矩阵调整边粗细）
for i in range(len(fused)):
    for j in range(i + 1, len(fused)):
        if fused[i, j] > 0.5:  # 设定一个相似性阈值
            G.add_edge(i, j, weight=fused[i, j])

# 输出网络中边的基本信息
print("\nNetwork Edges:")
print(G.edges(data=True))

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
